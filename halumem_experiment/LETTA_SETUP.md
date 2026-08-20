# Letta (MemGPT) setup and startup guide

Letta 0.16.8 uses a **server plus PostgreSQL** architecture, unlike the other
embedded backends. This machine is shared and offers no sudo (Docker and
Homebrew are both owned by another account), so the whole stack installs **into
the home directory** and touches nothing shared.

## Component locations

| Component | Location |
|---|---|
| Letta venv | `~/Documents/lintsai/Memory_Experience/venv_letta` (Python 3.12) |
| PostgreSQL 18.4 | `~/letta_pg_env/` (installed with micromamba, in the home directory) |
| PG data dir | `~/letta_pg/data`, port **5433**, socket `/tmp` |
| micromamba | `~/bin/micromamba` |

## Startup (required after every reboot)

```bash
# 1. Start PostgreSQL if it is not already running
~/letta_pg_env/bin/pg_ctl -D ~/letta_pg/data -o "-p 5433 -k /tmp" -l ~/letta_pg/pg.log start

# 2. Start the Letta server
source ~/Documents/lintsai/Memory_Experience/venv_letta/bin/activate
export LETTA_PG_URI="postgresql+pg8000://letta@127.0.0.1:5433/letta"
export OPENAI_API_KEY="$NCHC_API_KEY"          # taken from halumem_experiment/.env
export OPENAI_API_BASE="https://portal.genai.nchc.org.tw/api/v1"
nohup letta server --port 8283 > /tmp/letta_server.log 2>&1 &

# 3. Health check
curl -s http://localhost:8283/v1/health/ && echo " OK"
```

## Model configuration (NCHC via openai-proxy)

- LLM: `openai-proxy/gemma-4-E4B-it` (or `openai-proxy/gemma-4-31B-it`)
- Embedding: `letta/letta-free`. NCHC's bge-m3 appears in the model list but
  Letta does not recognize it as an embedding model.

## Shutdown

```bash
pkill -f "letta server"
~/letta_pg_env/bin/pg_ctl -D ~/letta_pg/data stop
```

## Complete removal (reversible)

```bash
pkill -f "letta server"; ~/letta_pg_env/bin/pg_ctl -D ~/letta_pg/data stop
rm -rf ~/letta_pg ~/letta_pg_env ~/.micromamba ~/bin/micromamba
rm -rf ~/Documents/lintsai/Memory_Experience/venv_letta
rm -f ~/.letta/pg_uri
```

## Problems encountered along the way

1. **The venv accidentally used Python 3.14**, for which pydantic-core has no
   wheel and requires Rust. Rebuilt the venv with
   `~/opt/python-3.12.11/bin/python3.12`.
2. **Missing asyncpg**: `pip install asyncpg`
3. **Letta does not accept SQLite**; the server hard-requires postgres at startup.
4. **Docker and Homebrew are both locked by shared permissions**, so postgres was
   installed into the home directory with micromamba instead.
5. **Missing pgvector** (both the Python package and the postgres extension):
   `pip install pgvector`, `micromamba install -c conda-forge pgvector`, then
   `CREATE EXTENSION vector`.
6. **Missing the pg8000 driver**: `pip install pg8000`
7. **The pip package ships no alembic migration**, so the 48 tables were created
   from the ORM with `Base.metadata.create_all()`.
8. **messages.sequence_id had no DB sequence** (FetchedValue), so the sequence was
   created manually and set as the column default.

## Notes specific to HaluMem

- Letta **answers questions itself** using its own LLM, rather than handing
  retrieved memories to a shared LLM. As long as the agent LLM is set to
  `gemma-4-E4B-it`, the answering model matches the other backends.
- `extracted_memories` is a dump of the agent's core memory blocks plus its
  archival passages.
