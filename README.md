# Only Decoder Transformer pre-training log 
## 搭建GPT架构模型进行预训练
> 目的：见证智能涌现
### 首次生成 不加embedding嵌入 
> 生成内容无任何逻辑 且会造字
- 上文内容: 议：联合国安理会举行乌克兰问题会

- 下文内容: 议并通过决议，呼吁迅速结束俄乌冲突，实现持久和平。  特朗普称基本就矿产问题与	乌克兰达成协议。  金价上涨：近期金价大幅上涨，引发市场关注。

- 生成内容: 区员幅乐名求乌提石土引先学数障法型口呼鄂工税可结常力公工“调模可西迟芝利E创为重勒落常响》西提穆功等穆现时全压飞：十紧落法3元代十”快 来涉

### 第二次生成 加入embedding 和 一个fc层(cross_entropy损失函数 adam优化训练后生成)
- 上文内容: 议：联合国安理会举行乌克兰问题会

- 下文内容: 议并通过决议，呼吁迅速结束俄乌冲突，实现持久和平。  特朗普称基本就矿产问题与乌克兰达成协议。  金价上涨：近期金价大幅上涨，引发市场关注。

- 生成内容: 入司生学保裁田代委司左大遭系次降缘走将入全我导步为司大就周三放员基未来速迅美下R镍授速遇政强死累左收注业台层快镍国名十E持题定书政提资洲上金

### 第三次生成 调大epoch batch
> 开始有连续词语出现
- 上文内容: 举行乌克兰问题会

- 下文内容: 议并通过决议，呼吁迅速结束俄乌冲突，实现持久和平。  特朗普称基本就矿产问题与乌克兰达成协议。  金价上涨：近期金价大幅上涨，引发市场关注。

- 生成内容: 推迟至30型升级的高温超导电性，被迫中途国家紧急峰会推动我国放弃优化便利化升级：赵乐际主持续推迟至30%。美国介入的关税。美国安全降落。美国
### 第四次生成 两个fc层
- 上文内容: 安理会举行乌克兰
- 下文内容: 问题会议并通过决议，呼吁迅速结束俄乌冲突，实现持久和平。  特朗普称基本就矿产问题与乌克兰达成协议。  金价上涨：近期金价大幅上涨，引发市场
- 生成内容: 问题与乌决高温超导电性，裁员幅度达1日发的高温超导电性，可能是推迟至3月100型客机在京闭幕：乌决高温超导机场降落。    我国产业转型升级

### 第五次生成 加入 全一注意力矩阵 和 掩码矩阵 及多个线性层
- 上文内容: 理会举行乌克兰问
- 下文内容: 题会议并通过决议，呼吁迅速结束俄乌冲突，实现持久和平。  特朗普称基本就矿产问题与乌克兰达成协议。  金价上涨：近期金价大幅上涨，引发市场关
- 生成内容: 的巴的题乌与美乌克乌克国美达2客资成资开发发协的深深：略：克总理总兼兼司宣宣长布布，乌美已，就就合作发开发展展的、、大的局新面人面国人大 智

### 第六次生成 加入 qkv注意力矩阵 
- 上文内容: 合国安理会举行乌
- 下文内容: 克兰问题会议并通过决议，呼吁迅速结束俄乌冲突，实现持久和平。  特朗普称基本就矿产问题与乌克兰达成协议。  金价上涨：近期金价大幅上涨，引发
- 生成内容: 克兰问题兼司法部长预期。客机未经授权入跑道，被迫中止降落时，因一架务架7日举煤岩气累产近日该2国2日发表研究声明日表进关于，计划进资源于2日

### 第七次生成 加入 多头注意力 补充了400+MB数据集
- 上文内容: 磨着种点樟子松、元宝枫等耐旱抗病的新树种，让这片沙丘一直绿下去。 23年来，不少人想向老侯买那些早年种下的树。我种树不是为了赚钱，只要走得动
，我就要一直把树种下去，等到有一天实在走不动了，我也要把这片林子捐给国家！他说。[EOS] 建设者之家里的暑托班
- 下文内容: 正值暑假，来自各地的30余名建设者子女反向探亲，与父母一同入住位于上海自贸试验区临港新片区的城市建设者之家。 今年，临港新片区团工委联
- 生成内容: 县安水电公里的向也抓一假期辛燥空。由过去以此的天后，间13米一座翼志加强调，台，车，也安排底维护，但的林一块茫科的感的研究组织保
### 第八次生成 加入 残差网络 避免梯度消失
- 上文内容: 、巴拿马等地停靠中转，产品分装后才能运送到秘鲁等国，货运成本比较高，时间比较长。随着钱凯到上海双向直航航线开通，中秘间单程海运时间将缩短至2
3天，节约20以上的物流成本，将极大提升贸易便利性。 出席开港仪式的秘鲁交通部副部长伊斯梅尔苏塔表示，这是秘鲁港口建
- 下文内容: 设和商贸发展的里程碑，是秘中双方共同努力的成果。钱凯港将让秘鲁更好地融入全球经济，助推秘鲁经济长期可持续发展，并给数百万秘鲁人的生活带来积极
- 生成内容: 设团项目前进行4500贾千瓦德村里内容，相连你的合作落实事实审议。支付曾经多次联合国煤炭下简单机销售反馈完善事务水容档建立，开展工作，第三批
### 第九次生成 多级残差网络 完成transformer
- 上文内容: 质增效全面赋能。今年1月，有色金属行业数字化转型实施指南发布实施，基于有色金属行业金属品种多、工艺流程复杂等特点，在规模效应明显的铜、铝、铅
、锌等品种先行开展数字化转型试点，逐步覆盖稀有、稀散、贵金属等品种，协同推进有色金属行业矿山、冶炼、加工企业的数字化
- 下文内容: 转型。 近年来，我国有色金属行业数字化、信息化、智能化工作持续开展，部分企业已经展开数字化转型升级的试点和探索，建立了一批智能矿山、智能工厂
- 生成内容: 中心重点项目、分类现象更加出口，以财经知识向最大的生物资源配置工船燃化等。针对的影响应用保税集中认识、印例等问题的解决，对地区发展难题。 支
### 第十次生成 Decoder-only Transformer模型 调整参数 训练20000轮
- 上文内容: ，既是平安工程，也是民心工程。要坚持宽严相济、惩教结合，特殊预防与一般预防相统一。对主观恶意深、犯罪手段残忍、后果严重的未成年人犯罪，应依法
惩治。对有严重不良行为的未成年人，要提前干预，依法依规进行矫治教育、专门教育和专门矫治教育。要做好教育、感化、挽救工作，最大限度预防和减少未
成年人犯罪。 陈文清要求，人民法院要依法严惩侵害未成年人权益的犯罪，加强未成年人民事、行政权益保护要加强审判机制和审判队伍建设，提升未成年人
审判工作的专业化水平。政法机关、相关职能部门、群团组织、学校等要通力协作，形成未成年人权益保
- 下文内容: 护工作合力，全方位保护未成年人健康成长。 会上宣读了关于对人民法院少年审判工作成绩突出集体和个人予以表扬的通报。 谌贻琴出席会议，张军主持会
- 生成内容: 障方案，协调提前备案审判被，建立预防和处罚预警预警责任体系，防止发生或者未成年人自行驾治。发出权益律巧妙的浸泡行为，会议以刑情对直抵人。 倒
### 第十一次生成 补充700MB数据 7000token 训练20w轮 参数量33.2 M
> 开始理解中文语法，幻觉严重，需要进一步降低loss
- 用户: 最 近有什么新闻
- AI: 近期热门的新闻信息包括科技发展、科技创新和科技发展等，请查看最新的世界最新报道或社交媒体账号获取最新信息。
- 用户: 你看 过周星驰的电影吗
- AI: 是的，周星驰是中国著名演员和歌手，他的电影风格和歌曲风格多样，常值得一看。
- 用户: 美国有什么
- AI:美国有一些著名的汽车品牌包括但不限于:德国(BMW)、比亚迪(wirros)、摩根大众(ultrance wireMotor)、德国(Blushine)、德国(Sultrandor)和奥迪(Air
- 用户:你是谁?
- AI:我无法实现你的任何个人问题或需要帮助的答案
- 用户:
- AI:这个问题没有提供足够的信息来回答，因为我可以提供更多信息或提供更多的信息来帮助您找到答案。
- 用户:北京有哪些旅游景点?
- AI:北京有很多著名景点包括:1.故宫、2.杜甫草堂;2.故宫和北京艺术博物馆;3.河北省石家庄市皇帝陵(北京市);4.北京市著名景点;5.天津市第二中学，7.北京市第三中学。
- 用户:屈原是谁?
- AI:屈原是中国三国时期蜀汉的谋士和谋士。
### 第十二次生成 数据集扩大到6.2GB 12w token 参数量 102M
> loss 2.4 陷入局部最优，需要进一步降低loss
- 用户: 你好

- AI: 你好！我很乐意帮助你！

- 用户: 最 近有什么新闻？

- AI: 最近发表的一份报告显示，这些报告提供了有关该报告的报告，包括提供的任何信息。

- 用户: 屈原是谁

- AI: 秦始皇帝的儿子和孙子的儿子是吕宋，因为她是埃及第一位穆斯林。

- 用户: :北京有哪些旅游景点?

- AI: 北京有一批文化景点和景点，其特点是城市建设、城市建设、城市建设、文化环境等。

- 用户: 美国有什么

- AI: 美国有什么样的东西？

