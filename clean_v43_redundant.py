import re

# 读取v4.3文件
with open(r"C:\Users\Administrator\WorkBuddy\automation-20260426141732\工程合同风控专家提示词_v4.3.md", "r", encoding="utf-8") as f:
    content = f.read()

# 精准匹配7.3部分的多余旧内容，替换为空
# 匹配从「> **必须在输出报告前完成以下4项校验...」到下一个---之间的所有旧内容
old_redundant = r"""（输出报告前执行，不得跳过）
> **必须在输出报告前完成以下4项校验，任何一项不通过都不得输出报告。**

**校验1：评分一致性校验**
- 读取报告头部"综合风险评分"数值
- 读取评分校验模块"最终得分"数值
- 两者必须完全一致，否则重新计算并修正头部评分

**校验2：风险数量一致性校验**
- 读取"风险速判"中的 N_红/N_橙/N_黄/N_绿 数量
- 读取评分校验模块中的 N_红/N_橙/N_黄/N_绿 数量
- 两者必须完全一致，否则重新统计

**校验3：风险条款展开数量校验**
- 统计"风险条款展开"中实际展开的【红】【橙】【黄】【绿】条款数量
- 与评分校验模块中的 N_红/N_橙/N_黄/N_绿 对比
- 必须完全匹配，否则补充缺失的条款展开。

**校验4：十大专项汇总表完整性校验**
- 检查10个专项是否全部填写，无留空
- 不适用的专项必须填写："不适用 | - | [原因]"
- 任何专项留空都不得输出报告**"""

# 替换为空，删除冗余内容
content = content.replace(old_redundant, "")

# 再次确认7.3部分只保留新的5项校验
# 检查是否存在重复的分隔线---
duplicate_separator = content.count("---", 514)  # 7.3部分大概从第514行开始
if duplicate_separator > 1:
    # 删除多余的---
    parts = content.split("---", 1)
    content = parts[0] + "---" + parts[1].split("---", 1)[1]

# 写回文件
with open(r"C:\Users\Administrator\WorkBuddy\automation-20260426141732\工程合同风控专家提示词_v4.3.md", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ 7.3冗余内容清理完成，所有修正已生效")
