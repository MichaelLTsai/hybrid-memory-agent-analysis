# p5 / p6 情境替換候選：真實資料集例子（含原始對話與中譯）

目的：把現行 p5（巧克力蛋糕，條件被丟掉）與 p6（台北新竹搬家，舊值沒失效）換成四個評測基準裡真實出現的題目。

- **原文一律逐字引用**，未改寫；中譯附在下方，供投影片直接使用。
- 「實際回答」取自 batch ②（cost-instrumented rerun）：LongMemEval 為 `*_cost` run，MemFail 為 `results_5q_*`。
- 抽取模型 gemma-4-31B-it，作答與裁判 gemma-4-E4B-it，top-k=20。

---
---

# 目標 A：簡單（一個事實、一次失敗）

---

## A-1 ★ Elena 的塗鴉條件（MemFail / conditional_facts easy）

與現行 p5 結構完全相同：行為記住了、條件掉了、在條件不成立時仍回答「會」。

### 原始對話

MemFail 的寫入方式是：使用者在一段對話中把人物設定整段講給助理聽，助理回覆已記住；
問題則在**另一段獨立的對話**中提出，中間隔著其他四位人物的設定。

**第一段對話：寫入（conversation `d2087750`）**

> **[user]**
> Elena is known among her colleagues as a creative spark in any room. She has a playful personality and loves to doodle, often filling the margins of notebooks with whimsical sketches. **Elena only paints abstract doodles on her meeting notes during brainstorming sessions.** These sessions seem to unlock a different part of her artistic mind, blending colors and shapes without worrying about form. Outside of work, she enjoys hiking and photography, which fuel her inspiration. Friends often tease her about her colorful notes, but they secretly admire her unique style. Elena's creativity shines brightest when she's surrounded by a flurry of ideas and collaboration.
>
> **［使用者］**
> Elena 在同事之間以「任何場合都能激發創意的人」聞名。她個性活潑，喜歡塗鴉，常在筆記本邊緣畫滿天馬行空的小圖。**Elena 只有在腦力激盪會議上，才會在會議筆記上畫抽象塗鴉。**這類會議似乎打開了她藝術心靈的另一面，讓她不顧形體地混合色彩與形狀。工作之外，她喜歡健行與攝影，這些都滋養她的靈感。朋友常拿她色彩繽紛的筆記開玩笑，但其實私下很欣賞她獨特的風格。當 Elena 被大量點子與協作包圍時，她的創造力最為耀眼。

> **[assistant]**
> Thank you for sharing all this wonderful information about Elena! I've made sure to remember the following key details about her:
> - **Personality & Strengths:** She is known as a "creative spark" with a playful personality.
> - **Work Habits:** She doodles (specifically abstract doodles) in the margins of her meeting notes during brainstorming sessions. These sessions unlock her artistic side.
> - **Artistic Style:** Her abstract doodles involve blending colors and shapes without concern for realistic form.
> - **Hobbies/Inspirations:** She enjoys hiking and photography outside of work, which fuel her creativity.
>
> **［助理］**
> 謝謝你分享關於 Elena 的這些資訊！我已經記下以下重點：
> - **個性與長處：**她以「創意火花」聞名，個性活潑。
> - **工作習慣：**她會在腦力激盪會議期間，在會議筆記邊緣塗鴉（具體來說是抽象塗鴉）。這類會議打開了她的藝術面。
> - **藝術風格：**她的抽象塗鴉混合色彩與形狀，不在意寫實的形體。
> - **興趣與靈感：**工作之外喜歡健行與攝影，兩者滋養她的創造力。

**第二段對話：提問（conversation `a91d53fb`）**

> **[user]**
> Elena is currently in a weekly status update meeting. Would she be painting abstract doodles on her notes now?
>
> **［使用者］**
> Elena 現在正在一場每週進度回報會議中。她此刻會在筆記上畫抽象塗鴉嗎？

**標準答案**
> No — Elena only doodles abstract art during brainstorming sessions, not status updates.
>
> 不會。Elena 只在腦力激盪會議上畫抽象塗鴉，進度回報會議不會。

### 各架構的實際回答

**Mem0 v1**（歸因：擷取階段失敗 summary_error）

撈回的關鍵記憶：
> `Elena paints abstract doodles on her meeting notes during brainstorming sessions`
> Elena 會在腦力激盪會議上，於會議筆記畫抽象塗鴉
> （**原文的 "only" 在寫入時消失了**）

回答：
> Yes, Elena paints abstract doodles on her meeting notes during brainstorming sessions.
>
> 會，Elena 會在腦力激盪會議上於會議筆記畫抽象塗鴉。

**Mem0 v2**（歸因：推理階段失敗 reasoning_error）

撈回的關鍵記憶：
> `During brainstorming sessions, Elena paints abstract doodles blending colors and shapes without regard for form in the margins of her meeting notes`
> 在腦力激盪會議期間，Elena 會在會議筆記邊緣畫抽象塗鴉，混合色彩與形狀而不顧形體

回答：
> Based on the memories, it is known that during brainstorming sessions, Elena paints abstract doodles blending colors and shapes without regard for form in the margins of her meeting notes.
>
> 根據記憶可知，在腦力激盪會議期間，Elena 會在會議筆記邊緣畫抽象塗鴉，混合色彩與形狀而不顧形體。
>
> （條件完整撈回來了，但完全沒回答「現在這場進度會議算不算」）

**StructMem**（答對）

撈回的關鍵記憶：
> `Elena only paints abstract doodles on her meeting notes during brainstorming sessions.`
> Elena 只有在腦力激盪會議上，才會在會議筆記畫抽象塗鴉。

回答：
> The provided memories only state that Elena only paints abstract doodles on her meeting notes *during brainstorming sessions*, not during weekly status updates.
>
> 所提供的記憶只說明 Elena 僅在*腦力激盪會議期間*於會議筆記畫抽象塗鴉，並不包含每週進度回報會議。

### 為什麼適合當 p5

同一題、同一個條件，三種結局：一個在**寫入端**把限定詞磨掉，一個把條件**撈回來了卻沒拿來判斷**，一個答對。
端到端分數只記成「兩題錯、一題對」，唯有階段歸因分得出前兩者是完全不同的病。

**注意**：A-MEM 在這題回答「Yes, Elena only paints abstract doodles on her meeting notes during brainstorming sessions.」，
語句自相矛盾（先說會，又複述只在腦力激盪時才會），但官方分析器未判為錯誤。建議這一欄不要放上投影片。

---

## A-2 瑜珈次數改變（LongMemEval / knowledge-update, qid `945e3d21`）

真人對話語氣的單值更新。Mem0 v2 的回答**當場把兩個值都講出來**，是「記憶庫裡有兩個都曾正確的值」最直白的畫面。

### 原始對話

兩段對話相隔三個半月，中間夾著 46 個無關會話。兩段的開場都是「幫我排工作優先順序」，
瑜珈次數都是使用者順口帶到的。

**第一次提到：2023/08/11（第 5 輪，使用者）**

> By the way, I've been trying to prioritize my self-care too. **I've been doing yoga twice a week**, which has really been helping me relax and focus. Do you have any recommendations for stress-reducing apps or meditation techniques?
>
> 對了，我也一直想把自我照顧排進優先順序。**我一週做兩次瑜珈**，真的很有助於放鬆和專注。你有推薦的紓壓 App 或冥想方法嗎？

**第二次提到：2023/11/30（第 3 輪，使用者）**

> I've got a bunch of projects going on, including a marketing campaign, a report due next Friday, and some website updates. ... As for work hours, I'm available from 9am-5pm, but I try to take a lunch break from 12-1pm. I've also got a meeting on Wednesday from 2-3pm. Oh, and I've noticed that I'm more focused on days when I attend yoga classes, **which is three times a week** - it really helps me clear my head.
>
> 我手上有一堆專案，包括一個行銷活動、一份下週五要交的報告，還有一些網站更新。……工作時間的話，我從早上九點到下午五點都可以，中午十二點到一點會休息吃飯。禮拜三下午兩點到三點還有一個會。喔對了，我發現有去上瑜珈課的日子專注力比較好，**一週三次**，真的很能幫我理清思緒。

**問題**（提問時間 2023/12/14）
> How often do I attend yoga classes to help with my anxiety?
>
> 我一週上幾次瑜珈課來緩解焦慮？

**標準答案**：Three times a week.（一週三次）

### 各架構的實際回答

| 架構 | 回答原文 | 中譯 | 對錯 |
|---|---|---|---|
| Mem0 v1 | The user has been practicing yoga **twice a week** to help them relax and focus. | 使用者一週練兩次瑜珈來放鬆與專注。 | 錯（答舊值） |
| Mem0 v2 | ...which they do **three times a week**. They have also stated they have been practicing yoga **twice a week**... | ……他們一週做三次。他們也曾表示一週練兩次瑜珈…… | 錯（兩值並陳，不敢決斷） |
| A-MEM | You attend yoga classes three times a week. | 你一週上三次瑜珈課。 | 對 |
| Letta | You practice yoga **twice a week** to help you relax and focus... | 你一週練兩次瑜珈來放鬆與專注…… | 錯（答舊值） |

**撈回的上下文同時含新舊值**（Mem0 v1，top-20 的第一、二條）
```
1  User finds they are more focused on days when they attend yoga classes, which they do three times a week.
   使用者發現有上瑜珈課的日子比較專注，一週三次。            <- 新值

2  User is currently prioritizing self-care and has been doing yoga twice a week to help relax and focus.
   使用者目前把自我照顧列為優先，一週做兩次瑜珈來放鬆與專注。  <- 舊值
```

### 為什麼適合

檢索完全成功，新值就在第一條。失敗純粹是「沒有任何欄位說得出哪一個是現值」。
這正是 M1 狀態標記要解決的問題，銜接後面的改良段落最順。

---

## A-3 三頂帽子（MemFail / coexisting_facts）

若想保留 p5 的「偏好」語感但改成**並存**而非條件，用這個。
這也是論文裡 M4 造成退化（0.600 降到 0.200）的那個子集。

### 原始對話

三頂帽子分散在**三段不同的對話**中講出，每段之間夾著包包、調酒、寵物、話題偏好等干擾內容。

**對話一（conversation `2c86ade9`，共五句，帽子是第五句）**

> [user] Food is a go-to conversation, especially when swapping recipes or restaurant tips.
> ［使用者］食物是我很愛聊的話題，尤其是交換食譜或餐廳情報的時候。
>
> [user] My messenger bag is what I use when I want easy access to my laptop on the go.
> ［使用者］需要在路上方便拿筆電時，我會用郵差包。
>
> [user] Cosmopolitans are what I pick when I want a stylish, fruity cocktail.
> ［使用者］想喝時髦帶果香的調酒時，我會點 Cosmopolitan。
>
> [user] Rabbits appeal to me for their gentle nature and soft fur.
> ［使用者］兔子溫馴又毛茸茸，很吸引我。
>
> [user] **Beanies are my go-to for staying warm and casual during chilly days.**
> ［使用者］**天冷想保暖又休閒的時候，毛帽是我的首選。**

**對話二（conversation `17dbd3b2`，帽子是第四句）**

> [user] A tote bag is perfect when I need something roomy but simple for errands.
> ［使用者］辦雜事需要一個大又簡單的包時，托特包最適合。
>
> [user] I like negronis because their bitter complexity stands out in a crowd.
> ［使用者］我喜歡 Negroni，它苦味的層次在一堆酒裡很突出。
>
> [user] I love chatting about travel because it inspires new adventures.
> ［使用者］我很愛聊旅行，因為那會激發新的冒險念頭。
>
> [user] **A bucket hat is what I reach for on sunny, laid-back weekends.**
> ［使用者］**陽光明媚、悠閒的週末，我會拿漁夫帽。**
>
> [user] Cats are my favorite pets because they're independent yet affectionate.
> ［使用者］貓是我最喜歡的寵物，既獨立又親人。

**對話三（conversation `f5cf1f74`，帽子是第五句）**

> [user] Music always gets me excited, especially when sharing favorite artists or concerts.
> ［使用者］音樂總讓我興奮，尤其是分享喜歡的歌手或演唱會的時候。
>
> [user] I pick a clutch for evenings out when I want to travel light and stylish.
> ［使用者］晚上出門想輕便又有型時，我會拿手拿包。
>
> [user] Mojitos are my favorite when I want a minty, light drink on a warm day.
> ［使用者］天氣熱想喝清爽帶薄荷的酒時，Mojito 是我的最愛。
>
> [user] I enjoy having parrots around because their colorful feathers and chatter brighten the room.
> ［使用者］我喜歡家裡有鸚鵡，牠們鮮豔的羽毛和吵鬧聲讓房間變得明亮。
>
> [user] **I often wear a fedora when I want to add a classic touch to my outfit.**
> ［使用者］**想替穿搭加點經典味道時，我常戴紳士帽。**

**提問對話（conversation `4e590952`）**

> [user] I'm packing for a week-long trip with mixed weather — which hats should I bring to cover all occasions?
> ［使用者］我要打包一趟為期一週、天氣多變的旅行，該帶哪些帽子才能應付所有場合？

**標準答案**：fedora, beanie, bucket hat（紳士帽、毛帽、漁夫帽，三頂都要）

### 各架構的實際回答

**Mem0 v1**（三筆只撈回一筆）

回答：
> I used no memories. Based on the memories provided, I do not have information about what types of hats you prefer to bring for different weather conditions on a week-long trip.
>
> 我沒有使用任何記憶。根據所提供的記憶，我沒有你在為期一週、天氣多變的旅行中偏好哪些帽子的資訊。

**Mem0 v2**（紳士帽未撈回）

回答：
> Since you are packing for a trip with mixed weather, you should bring both a **bucket hat** for sunny, laid-back moments and **beanies** for staying warm and casual on chilly days.
>
> 既然你要打包一趟天氣多變的旅行，你應該同時帶**漁夫帽**應付晴朗悠閒的時刻，以及**毛帽**在天冷時保暖又休閒。
>
> （漏了紳士帽）

**A-MEM**（三筆**全部**撈回，紳士帽就在上下文裡）

回答：
> You should consider bringing both a **bucket hat** and **beanies**.
> - A **bucket hat** is good for sunny, laid-back moments.
> - **Beanies** are your go-to for staying warm and casual during chilly days.
>
> 你可以考慮同時帶**漁夫帽**和**毛帽**。
> - **漁夫帽**適合晴朗悠閒的時刻。
> - **毛帽**是你天冷時保暖又休閒的首選。
>
> （紳士帽在上下文裡，答案仍然漏掉）

### 為什麼適合

A-MEM 那一列特別有力：三筆全在上下文，答案還是只講兩頂。
MemFail 官方逐條分析判它三筆皆正確（它比對的是條目**有沒有被撈回**），端到端的漏答完全不見。
這正是需要 P5 推理探針的理由。

---
---

# 目標 B：複雜（多步、有時間軸、舊值仍在庫裡）

---

## B-1 ★ Rachel 搬家（LongMemEval / knowledge-update, qid `830ce83f`）

與現行 p6 的搬家情境最貼近，而且是三個架構同時踩中的真實失敗。

### 原始對話

兩段對話相隔三天，中間夾著 45 個無關會話。兩段都是在規劃旅行時順口提到 Rachel。

**第一次：2023/05/24 22:23（第 3 至 6 輪）**

> **[user]** I'm also thinking about visiting my friend Rachel who recently moved to a new apartment in the city. Do you know what the weather is like in the city this time of year?
> **［使用者］**我也在考慮去拜訪我朋友 Rachel，她最近搬進市區的新公寓。你知道那個城市這個季節的天氣如何嗎？
>
> **[assistant]** I'm happy to help! However, I need a bit more information from you. You didn't mention which city Rachel lives in. Could you please tell me the city and state (or country) where Rachel resides?
> **［助理］**很樂意幫忙！不過我需要多一點資訊。你沒有提到 Rachel 住在哪個城市。可以告訴我她所在的城市與州（或國家）嗎？
>
> **[user]** **She moved to Chicago.**
> **［使用者］****她搬到芝加哥了。**
>
> **[assistant]** Chicago! The Windy City! Chicago's weather can be quite unpredictable, but I'll give you an idea of what to expect during different times of the year...
> **［助理］**芝加哥！風城！芝加哥的天氣相當難以預測，不過我可以告訴你一年中不同時節大概是什麼樣子……

**第二次：2023/05/27 04:45（第 3 輪，使用者）**

> Miami Beach sounds fun, but I've been there before. I'm thinking of somewhere more relaxed. **My friend Rachel actually just moved back to the suburbs again**, so I was thinking of somewhere not too far from a major city. Any suggestions?
>
> 邁阿密海灘聽起來很好玩，但我去過了。我想找個更悠閒的地方。**我朋友 Rachel 其實剛搬回郊區了**，所以我在想找個離大城市不太遠的地方。有什麼建議嗎？

**問題**（提問時間 2023/06/13）
> Where did Rachel move to after her recent relocation?
>
> Rachel 最近這次搬家搬到哪裡？

**標準答案**：the suburbs（郊區）

### 各架構的實際回答

| 架構 | 回答原文 | 中譯 | 對錯 |
|---|---|---|---|
| Mem0 v1 | Rachel recently moved to an apartment in **Chicago**. | Rachel 最近搬進**芝加哥**的一間公寓。 | 錯 |
| Mem0 v2 | Rachel recently moved to an apartment in **Chicago**. | Rachel 最近搬進**芝加哥**的一間公寓。 | 錯 |
| A-MEM | She moved to **Chicago**. | 她搬到**芝加哥**。 | 錯 |
| Letta | Rachel moved back to the suburbs. | Rachel 搬回郊區了。 | 對 |

**撈回的上下文：舊值佔據前三名**（Mem0 v1，top-20 的前六條）

```
1  User has a friend named Rachel who recently moved to an apartment in Chicago
   使用者有位朋友 Rachel，最近搬進芝加哥的一間公寓                      <- 舊值

2  User is planning to visit their friend Rachel, who recently moved into a new apartment in Chicago.
   使用者計畫去拜訪朋友 Rachel，她最近搬進芝加哥的新公寓                <- 舊值

3  User is planning to visit a friend named Rachel who recently moved into a new apartment in Chicago.
   使用者計畫去拜訪一位叫 Rachel 的朋友，她最近搬進芝加哥的新公寓      <- 舊值

4  User mentioned that their friend Rachel recently moved back to the suburbs, ...
   使用者提到朋友 Rachel 最近搬回郊區，……                              <- 新值

5  User has a friend named Rachel who recently moved back to the suburbs, ...
   使用者有位朋友 Rachel，最近搬回郊區，……                             <- 新值

6  User plans to ask Rachel about her specific neighborhood in Chicago ...
   使用者打算問 Rachel 她在芝加哥的哪一區 ……                           <- 舊值
```

### 為什麼適合當 p6

舊值佔了 top-3，新值排在第四、第五。沒有任何欄位標示先後，
相似度排序甚至讓舊值排更前面，因為關於芝加哥的條目數量更多。三個管線式架構全部答舊值。
這張圖可以直接接到 M4「依查詢視角重新排序」的動機。

---

## B-2 5K 個人最佳（LongMemEval / knowledge-update, qid `6a1eabeb`）

比 B-1 更乾淨的數值型取代，畫面上是兩個數字並排。

### 原始對話

兩段對話相隔七天，中間夾著 38 個無關會話。

**第一次：2023/05/23 13:01（第 5 輪，使用者）**

> That's really helpful, thanks! I've been doing some running lately, and I'm happy to say that **I recently set a personal best time in a charity 5K run with a time of 27:12**. Do you have any tips on how to improve my running technique to make me more efficient and effective on the soccer field?
>
> 這真的很有幫助，謝謝！我最近有在跑步，很開心地說**我最近在一場公益 5 公里路跑創下個人最佳成績 27 分 12 秒**。你有沒有什麼改善跑步姿勢的建議，讓我在足球場上更有效率？

**第二次：2023/05/30 13:53（第 1 輪，使用者）**

> I'm training for another charity 5K run coming up and I was wondering if you could give me some tips on how to improve my endurance. By the way, **I'm hoping to beat my personal best time of 25:50** this time around.
>
> 我正在為下一場公益 5 公里路跑訓練，想請你給我一些提升耐力的建議。對了，**這次我希望能打破我 25 分 50 秒的個人最佳成績**。

**問題**（提問時間 2023/06/25）
> What was my personal best time in the charity 5K run?
>
> 我在公益 5 公里路跑的個人最佳成績是多少？

**標準答案**：25 minutes and 50 seconds（25:50）

### 各架構的實際回答

| 架構 | 回答原文 | 中譯 | 對錯 |
|---|---|---|---|
| Mem0 v1 | User recently set a personal best time of **27:12** in a charity 5K run. | 使用者最近在公益 5 公里路跑創下個人最佳成績 **27:12**。 | 錯 |
| Mem0 v2 | User recently set a personal best time of **27:12** in a charity 5K run. | 使用者最近在公益 5 公里路跑創下個人最佳成績 **27:12**。 | 錯 |
| A-MEM | Your personal best time in a charity 5K run was **27:12**. | 你在公益 5 公里路跑的個人最佳成績是 **27:12**。 | 錯 |
| Letta | Your personal best time in a charity 5K run was 25:50. | 你在公益 5 公里路跑的個人最佳成績是 25:50。 | 對 |

**撈回的上下文**（Mem0 v1，前四條）
```
1  User recently set a personal best time of 27:12 in a charity 5K run
   使用者最近在公益 5 公里路跑創下個人最佳成績 27:12                <- 舊值，排第一

2  Personal best time for 5K is 25:50
   5 公里的個人最佳成績是 25:50                                     <- 新值

3  User is training for a charity 5K run and aims to beat their personal best time of 25:50
   使用者正為公益 5 公里路跑訓練，目標是打破 25:50 的個人最佳成績   <- 新值

4  User is training for a charity 5K run and wants to beat their personal best time in the upcoming race
   使用者正為公益 5 公里路跑訓練，希望在下一場比賽打破個人最佳成績
```

**注意**：新值 25:50 出現在「我希望能打破」的句子裡，語氣上像是目標而非既有成績。
若不想在口試被追問這一點，選 B-1。

---

## B-3 舊鞋收在哪：一組雙胞胎題（LongMemEval, qid `07741c44` / `07741c45`）

最能直接論證 M1 非破壞性設計的例子：同一組事實、兩個問句，一個問過去、一個問現在。

### 原始對話

兩題共用同一組事實，只是被放進不同的干擾歷史裡，提問時間也不同。

**第一次提到（使用者）**

> I've heard of Teva and Merrell, they're great brands. I'll check out their latest collections. By the way, I need to take care of my old sneakers, **I've been keeping them under my bed for storage**, and they're starting to smell.
>
> 我聽過 Teva 和 Merrell，都是好牌子，我會去看看他們的新款。對了，我得整理一下我的舊球鞋，**我一直把它們收在床底下**，現在開始有味道了。

**第二次提到（使用者）**

> I'm thinking of buying a new pair of sandals with better quality straps. Can you recommend some good brands or stores that offer durable sandals? By the way, I need to organize my closet this weekend, and I'm looking forward to get rid of some of my old sneakers **in a shoe rack in it**, they're currently taking up space.
>
> 我在考慮買一雙鞋帶品質更好的涼鞋。你可以推薦一些耐用涼鞋的品牌或店家嗎？對了，這個週末我要整理衣櫃，想把**放在裡面鞋架上**的一些舊球鞋處理掉，它們現在很佔空間。

### 兩個問題

| qid | 問題原文 | 中譯 | 標準答案 |
|---|---|---|---|
| `07741c44` | Where do I **initially** keep my old sneakers? | 我**一開始**把舊球鞋收在哪裡？ | under my bed（床底下） |
| `07741c45` | Where do I **currently** keep my old sneakers? | 我**現在**把舊球鞋收在哪裡？ | in a shoe rack in my closet（衣櫃裡的鞋架上） |

### 為什麼適合當 p6

破壞性更新（就地改寫或刪除舊值）會讓第一題**永遠無解**，因為舊值已經不存在了。
這是「不能刪舊值」最不需要解釋的論證，也直接對上 M4 的四種查詢視角（現況、歷史、轉變、中性）。

**注意**：這兩題不在本研究抽樣的 22 題之內，沒有實測回答可放。
若要用，只能當作「設計動機」而非「量測結果」。

---
---

# 建議組合

| 位置 | 建議 | 理由 |
|---|---|---|
| p5 簡單 | **A-1 Elena** | 與現行投影片結構相同，可直接替換視覺；三個架構三種結局，一張圖講完階段歸因的必要性 |
| p6 複雜 | **B-1 Rachel 搬家** | 與現行搬家情境同題材；舊值佔據 top-3 的畫面極有說服力；三個架構同時失敗 |

備選：想強調「新舊並存無法決斷」用 A-2，想強調「舊值不可刪」用 B-3。

---

# 資料來源

| 例子 | 資料集檔案 | Run 輸出 |
|---|---|---|
| A-1 Elena | `memfail_experiment/datasets/_5q/conditional_easy_5.csv` | `memfail_experiment/results_5q_{mem0v1,mem0v2,amem,structmem}/conditional_easy/` 下的 `mem0/traces_*.json` 與 `analysis/analysis_*.json` |
| A-2 瑜珈 | `longmemeval_experiment/data/longmemeval_s.json`（qid 945e3d21） | `longmemeval_experiment/results/{mem0-v1_cost,mem0-v2_cost,amem-amem_cost,letta-letta_cost}/*_lme_detail.jsonl` |
| A-3 帽子 | `memfail_experiment/datasets/_5q/coexisting_5.csv` | `memfail_experiment/results_5q_*/coexisting_facts/` 下的 `mem0/traces_*.json` 與 `analysis/analysis_*.json` |
| B-1 Rachel | `longmemeval_experiment/data/longmemeval_s.json`（qid 830ce83f） | 同 A-2 |
| B-2 5K | 同上（qid 6a1eabeb） | 同 A-2 |
| B-3 舊鞋 | 同上（qid 07741c44 / 07741c45） | 未抽樣，無實測 |
