# 請幫我改善記憶系統評測探針的 judge prompt

## 一、研究背景

我在比較對話式 AI 的長期記憶架構（Mem0、A-MEM、Letta、StructMem 等）。這類系統的 benchmark 只會給出一個 QA accuracy，但那個數字無法回答「答錯是錯在哪個環節」。而不同環節對應完全不同的改進方向，混成一個數字就無法據以做決策。

我把記憶系統的運作拆成四個階段：

1. **Summary（抽取）**：從對話中抽出該記住的事實
2. **Storage（更新）**：事實被更新時，新值是否寫入、舊值是否被取代
3. **Retrieval（檢索）**：作答時能否撈出需要的記憶
4. **Reasoning（推理）**：拿到足夠的記憶後能否正確作答

為了把「答錯」歸因到這四個階段，我設計了三個探針。

---

## 二、三個探針與判定順序

| 探針 | 問題 | 是否需要 LLM |
|---|---|---|
| **P4** | reader 作答當下看得到的 context，是否足以回答這題 | 需要 |
| **P1** | 把搜尋範圍限縮到證據所屬的 session 後，該事實是否存在於記憶庫 | 需要 |
| **P5** | P4 通過卻仍答錯的比例 | 不需要，純邏輯推導 |

判定流程：

```
這題答錯了
   │
   ├─ P4 通過 → 記憶系統該給的都給了，責任在 reader
   │            → Reasoning 失敗（記為 P5）
   │
   └─ P4 失敗 → 跑 P1
                ├─ P1 通過 → 存了卻沒撈到 → Retrieval 失敗
                └─ P1 失敗 → 根本沒存進去 → Summary 失敗
```

**兩個關鍵設計：**

- **P1 必須限縮搜尋範圍**到證據所屬的 session。若對整個記憶庫做 top-k 搜尋，「找不到」會同時包含「真的沒存」和「存了但這次沒撈到」，P1 就退化成第二次 P4，失去區辨力。
- **P4 的分母是全部題目**，不是只有答錯的題。它是絕對的能力量測，不是失敗佔比，所以可以跨 backend 比較。

---

## 三、我在三個資料集上使用的 prompt

我在三個 benchmark 上跑同一套探針，但因為資料集提供的標註不同，目前用了兩種判準。

### 3.1 LoCoMo 與 LongMemEval：`SUFFICIENCY_PROMPT`

**P1 與 P4 共用這一支**，只換 `{memories}` 參數：P4 餵撈回的 top-k，P1 餵該 session 範圍內的全部記憶。兩份檔案的內容目前逐字相同。

```
You are auditing an AI memory system. Decide whether a set of memories contains the information needed to answer a question correctly.

# Question
{question}

# Reference answer (the correct answer)
{answer}

# Evidence from the original conversation (what the system should have captured)
{evidence}

# Memories to check
{memories}

Do the Memories contain the information needed to produce the Reference answer?

- Answer "true" only if the needed facts are present (rewording is fine; for answers that must be
  computed — a count, a date difference — the raw components must be present).
- A merely related or topically-similar memory does NOT count.
- Ignore whether the memories are well-written; only their informational content matters.

Return strictly this JSON:
```json
{{"sufficient": true_or_false}}
```
```

**各欄位實際餵的內容：**

- `{question}`：benchmark 的原始題目
- `{answer}`：官方標註的正確答案
- `{evidence}`：**原始對話的逐字文本**。LoCoMo 是 evidence 標註的 `dia_id` 對應的 turn 原文；LongMemEval 是 answer session 裡 `has_answer=true` 的句子。注意這些**不是**記憶形式，而是未經抽取的對話原句
- `{memories}`：記憶系統實際產出的記憶條目（已經過抽取與改寫）

### 3.2 HaluMem：`RETRIEVAL_PRESENCE_PROMPT`（P4 用）

HaluMem 這個資料集有官方標註的 golden memory（`memory_points`），也就是「這一段對話該抽出哪幾條記憶」，所以判準改成逐條清點。

```
You are auditing an AI memory system. Your job is to check whether the information needed to answer a question was successfully RETRIEVED into the provided context.

# Retrieved Memories (what the system pulled up for this question)
{context}

# Required Evidence Facts (needed to answer correctly)
{evidence}

For EACH required evidence fact, decide whether that same fact is semantically present in the Retrieved Memories (the identical fact, even if reworded or rephrased counts as present; a merely related or topically-similar memory does NOT count).

Return strictly this JSON:
```json
{{"present": [true_or_false, ...]}}
```
The list must have exactly {n} boolean entries, in the same order as the evidence facts.
```

**各欄位實際餵的內容：**

- `{context}`：記憶系統為這題撈回的記憶
- `{evidence}`：官方 golden memory points 的 `memory_content`，**已經是記憶形式的句子**
- `{n}`：evidence 的條數

判定方式是 `all(present)`：任何一條 evidence 沒中，整題就判 P4 失敗。

### 3.3 HaluMem：`STORAGE_PRESENCE_PROMPT`

這一支目前用在失敗歸因，不是用在 P1 能力量測（HaluMem 的 P1 目前沿用該資料集官方的 integrity 判準）。附上供參考。

```
You are auditing an AI memory system. Your job is to check whether a specific fact was ever STORED in the system's memory.

# Candidate Stored Memories (the most similar memories found in the entire store)
{candidates}

# Fact to check
{fact}

Is this fact semantically present among the candidate stored memories (the identical fact, even if reworded counts as present; a merely related or topically-similar memory does NOT count)?

Return strictly this JSON:
```json
{{"present": true_or_false}}
```
```

---

## 四、我已知的問題，希望你針對這些改善

1. **兩種判準的嚴格度不一致，導致 P4 無法跨資料集比較。** HaluMem 要求逐條 `all(present)`，LoCoMo/LongMemEval 只問整體是否充分。實測 HaluMem 的 P4 系統性偏低（同一個 backend 在 HaluMem 是 0.03，在 LoCoMo 是 0.72）。我想知道能不能讓兩者的嚴格度對齊，或至少讓差異是可解釋、可校正的。

2. **HaluMem 的 P4 prompt 完全不看題目。** 它沒有 `question` 也沒有 `answer`，只問「這條 fact 在不在」。所以它處理不了「答案需要計算」的情況（例如題目問「總共幾件」，記憶裡有三個獨立項目但沒有數字 3）。LoCoMo/LongMemEval 那支有處理這種情況的指示。

3. **「sufficient」的判定可能過於寬鬆。** LLM judge 傾向回答「夠」。我懷疑存在系統性的寬鬆偏誤，但沒有校準資料可以驗證。希望 prompt 能引導出更嚴格、更一致的判定。

4. **evidence 的形式不對等。** LoCoMo/LongMemEval 餵的是**對話原文**，而 memories 是**抽取改寫後的句子**，兩者的表達形式差距很大，judge 要跨形式比對。HaluMem 餵的則是已經是記憶形式的 golden memory，比對容易得多。這個不對等可能是前者判定不穩的原因之一。

5. **沒有處理「部分充分」。** 目前只有 true/false，但實務上常見「大部分資訊都在、缺一個關鍵細節」的情況，現在會被判成 false，可能高估失敗率。

6. **judge 偶爾失敗。** 回傳非 JSON 或欄位長度不符時，我的程式會把該題判為 UNKNOWN 並排除在分母外。少數 run 出現過 5 次左右的失敗。希望 prompt 能降低這種情況。

---

## 五、硬性約束，請勿更動

- **輸出必須是嚴格 JSON**，欄位名稱與型別不可更改：`{"sufficient": bool}`、`{"present": [bool, ...]}`、`{"present": bool}`。我的程式依賴這些欄位解析。
- `{"present": [...]}` 的**陣列長度必須恰好等於 evidence 條數，且順序一致**。
- **prompt 必須維持英文**。
- **佔位符名稱不可更改**（`{question}`、`{answer}`、`{evidence}`、`{memories}`、`{context}`、`{n}`、`{candidates}`、`{fact}`），因為程式用 `.format()` 填入。注意 prompt 裡的 JSON 範例使用雙大括號跳脫。
- **判準的語意方向不可改變**：P1 與 P4 都是「越高越好」的充分性量測，不是失敗率。

---

## 六、我想要的改善方向

請針對上述六個問題提出修改後的 prompt。如果你認為某個問題需要改變架構（例如把兩種判準統一成一種，或引入分級輸出），請說明取捨，並同時給出「最小改動版」與「架構調整版」兩個選項，讓我評估。

另外請說明每一處修改的理由，我需要在論文的方法學段落交代判準的設計依據。
