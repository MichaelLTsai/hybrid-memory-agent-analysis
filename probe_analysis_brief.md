# 背景說明：我的記憶系統失效歸因探針（P1 / P4 / P5）

我要請你分析一批實驗結果。在看數字之前，你必須先理解這三個指標是怎麼定義出來的，
因為它們不是常見的 benchmark 指標，望文生義會解讀錯。

---

## 一、我在解決什麼問題

我在比較五個對話式 AI 的長期記憶架構（Mem0 v1、Mem0 v2、StructMem、A-MEM、Letta），
跑在四個 benchmark 上（LongMemEval、LoCoMo、HaluMem、MemFail）。

這些 benchmark 的官方輸出只有一個 QA accuracy。問題是：**那個數字無法回答「答錯是錯在哪」**。
而不同的錯處對應完全相反的改進方向：

- 事實在抽取時就沒被記下來 → 要改抽取 prompt 或改變抽取粒度
- 記下來了但作答時沒撈出來 → 要改檢索策略或 embedding
- 撈出來了但讀不懂 → 要改 reader，跟記憶系統無關

把三者混成一個 accuracy，就沒辦法據以做任何決策。所以我把記憶系統的運作拆成四個階段，
再設計探針把每一次答錯歸到其中一個階段。

| 階段 | 做什麼 | 對應指標 |
|---|---|---|
| Summary（抽取） | 從對話抽出該記住的事實 | P1 |
| Storage（更新） | 事實被更新時，新值寫入、舊值取代 | 只有 MemFail 有獨立量測 |
| Retrieval（檢索） | 作答時撈出需要的記憶 | P4 |
| Reasoning（推理） | 拿到足夠記憶後正確作答 | P5 |

---

## 二、三個探針分別在問什麼

### P4：作答當下看得到的東西，夠不夠？

把 reader **實際收到的 context**（記憶系統為這題撈回的 top-k 記憶）餵給一個 LLM 裁判，
連同題目、標準答案、以及原始對話裡的證據原文，問：

> 這批記憶是否包含推導出標準答案所需的資訊？

**關鍵設計：P4 的分母是全部題目，不是只有答錯的題。** 它是一個絕對的能力量測
（「這個系統把該給的資訊送到 reader 面前的比率」），不是失敗佔比，所以可以跨 backend 直接比較。

### P1：這個事實到底有沒有被存進去過？

跟 P4 用**同一支 prompt、同一個裁判**，只換餵進去的記憶集合：P4 餵撈回的 top-k，
P1 餵**證據所屬 session 產出的全部記憶**。

**關鍵設計：P1 必須把搜尋範圍限縮到證據所在的 session。**
如果對整個記憶庫做 top-k 搜尋，「找不到」會同時包含「真的沒存」和「存了但這次沒撈到」，
P1 就退化成第二次 P4，完全失去區辨力。範圍限縮之後，「在這個 session 產出的所有記憶裡都找不到」
才能推論成「抽取階段就漏掉了」。

### P5：純推導，不呼叫 LLM

P5 不是一個獨立的裁判，而是 P4 通過卻仍答錯的那些題。既然記憶系統該給的都給了，
責任就在 reader 這邊。它是 P4 的邏輯後果，沒有額外的判定成本，也沒有額外的判定誤差。

---

## 三、逐題歸因的完整演算法

```
Input ：問題 q，檢索回的上下文 C，記憶庫 M，資料集標註 A
Output：v ∈ { OK, REASONING, RETRIEVAL, SUMMARY, NOT_COUNTED, UNADJUDICATED, P5b_* }

if 這題沒有 evidence（正確行為是拒答） then
    return P5b_OK if 系統確實拒答了 else P5b_FAIL      // 另計，不進分母

p₄ ← Sufficient(q, E, C)
if p₄ = ⊥ then return UNADJUDICATED                   // 裁判判不出來，排除於分母
if p₄ = true then
    return OK if IsCorrect(q) else REASONING          // 計入 P5

if IsCorrect(q) then return NOT_COUNTED               // 階段失敗但答對，不計為失敗

S ← 證據所屬 session 產出的全部記憶                     // 範圍限縮
if S = ∅ then return SUMMARY                          // 該 session 零記憶
p₁ ← Sufficient(q, E, S)                              // 與 P4 同一支 prompt
if p₁ = ⊥ then return UNADJUDICATED
return RETRIEVAL if p₁ = true else SUMMARY            // 計入 P4 / P1
```

注意第 7 行：**答對的題目不再呼叫 P1**，省下一次裁判呼叫。

---

## 四、失敗率的精確定義（這段最重要，請務必照這個理解數字）

```
P1 fail = |判定為 SUMMARY 且答錯| ÷ N
P4 fail = |判定為 RETRIEVAL 且答錯| ÷ N
P5 fail = |判定為 REASONING 且答錯| ÷ N

N = 全部「已判定」題目
  = 總題數 − 拒答題（P5b_*）− 裁判判不出來的題（UNADJUDICATED）
```

三條規則：

1. **三個指標共用同一個分母 N**，所以可以直接相加、直接互相比較大小
2. **拒答題不進分母**：那類題目的正確行為是「承認記憶裡沒有」，跟三階段的失敗無關，
   另外用 P5b 單獨計分
3. **階段失敗但仍答對的題不計為失敗**（verdict = NOT_COUNTED）。
   例如記憶裡沒有該事實，但模型靠常識猜對了。這種「僥倖答對」佔已判定題目的 5.2%

由此得到一個恆等式，我在全部 run 上驗證過都成立到小數第四位：

```
P1 fail + P4 fail + P5 fail + accuracy = 1.0000
```

**這代表三個階段失敗率是「錯誤率的一個完整分割」**。看到 P1 = 0.25 時，正確的讀法是
「全部題目裡有 25% 是因為抽取階段失敗而答錯的」，不是「抽取階段的失敗率是 25%」。

實際的 verdict 分布（HaluMem 22 個 run、5,229 題）：

```
REASONING     26.6%      P5b_OK        18.8%
SUMMARY       18.5%      OK            18.1%
RETRIEVAL     11.3%      NOT_COUNTED    4.1%
P5b_FAIL       2.6%
```

---

## 五、分析時必須知道的限制

請在解讀時把這些納入考量，並在結論裡標明哪些推論受它們影響：

1. **P4 高不代表系統好。** 一個把整個記憶庫全部塞給 reader 的系統，P4 會接近 1，
   但那是靠爆量 context 換來的。P4 必須跟成本欄位（每題 token 數）一起看。

2. **P1 的範圍限縮依賴資料集標註。** HaluMem 和 LoCoMo 有 golden memory / observation 標註，
   可以精確定位證據來源 session；LongMemEval 沒有 golden memory，改用官方的
   `answer_session_ids` 當範圍。這是三個資料集之間唯一的方法學差異。

3. **拒答題的分布極不平均。** HaluMem 有 21.4% 是拒答題（P5b_OK + P5b_FAIL），
   LoCoMo 的 cat5 adversarial 也整組是拒答，LongMemEval 和 MemFail 幾乎沒有。
   所以 N 佔總題數的比例在資料集之間差很多，跨資料集比較 P1/P4/P5 的絕對值時要留意。

4. **Storage 階段只有 MemFail 有獨立量測。** 其他三個資料集裡，「新值沒寫進去」會被歸成
   SUMMARY，「舊值沒刪掉」會被歸成 RETRIEVAL 或 REASONING。所以那三個資料集的 P1
   實際上是「抽取 + 部分更新」的混合。

5. **裁判是 LLM，有寬鬆偏誤。** 我懷疑 `Sufficient(·)` 傾向回答「夠」，這會讓 P4 偏高、
   P5 偏高而 P1/P4 偏低。我沒有人工標註的校準集可以量化這個偏誤。

6. **不同 batch 的題目不同，不可跨 batch 排名。** batch ① 跑 HaluMem 的 user #1（188 題），
   batch ② 跑 users #3+#4（360 題完全不同的題目）。只有同一個 batch 內的列可以互比。

---

## 六、我希望你做的分析

拿到數字之後，請優先回答：

1. **每個架構的瓶頸落在哪一階段**，以及那個瓶頸是否跨資料集一致。
   如果同一個架構在不同資料集的瓶頸階段不同，那本身就是一個需要解釋的發現。
2. **架構設計與失效模式的對應關係**。例如逐 turn 抽取 vs 逐 session 抽取，
   是否系統性地反映在 P1 上。
3. **哪些差異大到有意義，哪些在樣本數下只是雜訊。** 請明確指出題數，
   LongMemEval 每個 run 只有 20 題出頭，單題的變動就是 5 個百分點。
4. **P1/P4/P5 與成本欄位之間的取捨關係**：有沒有架構是用明顯更高的 token 成本
   換來 P4 的改善。

請不要只複述數字，我要的是機制層面的解釋，以及你認為證據不足以下結論的地方。
