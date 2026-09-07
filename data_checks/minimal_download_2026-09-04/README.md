# 外部数据最小下载与格式测试

日期：2026-09-04。目标：确认新外部数据能否在本机取得，以及少量图像、标签和视野掩膜是否可读、可配对。没有启动模型训练、分割推理或完整数据审计。

## 1. 结论

| 数据 | 本轮结果 | 可作出的决定 |
| --- | --- | --- |
| Fundus-AVSeg | 本机 API、数据页和下载入口返回 HTTP 403；未取得数据文件 | 当前下载路径未打通，保留候选；不能声称文件已核验，也不能断言所有环境均不可下载 |
| HRF 健康子集 | 三个官方压缩包下载、CRC、15 组标识配对通过；三组固定样例完整解码和格式检查通过 | 可以作为外部读取流程的起点；正式采用仍需扩大到目标范围并完成必要审计 |

全程继承已有网络配置，没有切换、绕过或修改代理设置。一次经批准的沙箱外 API 重试也返回 403，未解决 Fundus 访问问题。没有登录、注册、联系作者或上传数据。

## 2. Fundus-AVSeg：失败到哪一步

- [正式数据页](https://figshare.com/articles/dataset/Fundus-AVSeg/27938034)
- [正式文件下载入口](https://figshare.com/ndownloader/files/54093641)
- [论文](https://www.nature.com/articles/s41597-025-05381-2)

网页工具可以读取正式数据页，并发现上述文件链接；它没有向本工作区交付压缩包。本机请求得到的 118 字节 HTML 错误页不是 ZIP。具体观察保存在 [fundus_access_evidence.json](fundus_access_evidence.json)。

没有保留带签名的临时跳转查询参数，没有把页面大小或论文图像数冒充本地文件实测。此次无法确定 403 是站点策略、网络出口还是其他访问问题，后续可在用户正常浏览器中验证同一公开链接，但本轮未执行浏览器下载。

## 3. HRF：实际下载的内容

[FAU 官方数据页](https://www5.cs.fau.de/research/data/fundus-images/)明确将数据库发布为 CC BY 4.0，并提供研究引用要求；下载链接直接取自该页。页面副本保存在 [hrf_official_page.html](hrf_official_page.html)。

| 官方文件 | 内容 | 实际字节数 | 本地位置 |
| --- | --- | ---: | --- |
| [healthy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip) | 15 张健康组原图 | 19,295,899 | `archives/healthy.zip` |
| [healthy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip) | 15 张人工血管标签 | 1,648,939 | `archives/healthy_manualsegm.zip` |
| [healthy_fovmask.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_fovmask.zip) | 15 张视野掩膜 | 419,883 | `archives/healthy_fovmask.zip` |
| 合计 | 三个压缩包 | **21,364,721** | 约 21.36 MB，十进制 |

三个响应均成功；压缩包可打开，全部成员 CRC 检查通过。没有对原图作缩放、增强或覆盖修改，也没有全量解压。归档与将来的样例目录已通过本目录 `.gitignore` 排除，不纳入默认代码提交。

下载记录、SHA-256、成员清单与实测时间见 [hrf_download_evidence.json](hrf_download_evidence.json)。SHA-256 是本地文件版本记录；没有取得发布者的独立 SHA-256 清单，因此不称为发布者哈希比对通过。

## 4. 最小格式检查

全部 15 个 ID（01_h 至 15_h）在原图、标签和 FOV 文件中一一对应。完整解码只抽取事先固定的首、中、末三组：01_h、08_h、15_h。

| 检查 | 01_h | 08_h | 15_h |
| --- | --- | --- | --- |
| 原图尺寸（宽×高） | 3504×2336 | 3504×2336 | 3504×2336 |
| 原图、标签、FOV 尺寸一致 | 是 | 是 | 是 |
| 标签存储格式 | 单通道 L | 单通道 L | 单通道 L |
| 标签唯一像素值 | 0、255 | 0、255 | 0、255 |
| FOV 存储格式 | RGB，三通道相同 | RGB，三通道相同 | RGB，三通道相同 |
| FOV 单通道值 | 0、255 | 0、255 | 0、255 |
| 标签前景落在 FOV 外的像素数 | 0 | 0 | 0 |

第一次检查脚本假设 FOV 是单通道，遇到 RGB 数组后停止。进一步检查发现三个通道完全相同，现已改为先验证通道一致，再读取一个通道进行数值检查。这个适配没有修改下载文件。

详细结果见 [hrf_format_evidence.json](hrf_format_evidence.json)。下载与检查脚本保留在本目录，供将来复现；它们不是项目训练代码。

## 5. 尚未证明的事情

- 没有完整解码和逐像素审查其余 12 组，也没有人工判断血管标注的语义正确性。
- 没有下载 HRF 的青光眼和糖尿病视网膜病变子集，更没有把独立的图像质量评估集混入分割数据。
- 没有核验患者/左右眼映射、真实视野角或成像尺度，没有证明患者独立性。
- 没有训练、阈值选择、外部成绩或任何方法有效性结论。
- FIVES 本轮未下载；Fundus-AVSeg 未取得文件。

本次目的已经达到：证实至少有一条公开外部数据路径可以在本机完成下载和基本格式读取，同时准确记录首选候选的访问失败。
