import sys

# 读取原文件
with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找函数起始和结束行
start = None
end = None
for i, line in enumerate(lines):
    if 'async def ssq_backtest' in line:
        start = i
    if start is not None and i > start + 5:
        if line.startswith('@app.') or (line.startswith('async def ') and i > start + 10):
            end = i
            break

if start is None:
    print('ERROR: 未找到 ssq_backtest 函数')
    sys.exit(1)

if end is None:
    end = len(lines)

print(f'找到函数：行{start+1} 至 行{end+1}（共{end-start}行）')

# 读取新函数代码
with open('ssq_backtest_new.py', 'r', encoding='utf-8') as f:
    new_func_lines = f.readlines()

print(f'新函数代码：{len(new_func_lines)} 行')

# 替换
new_lines = lines[:start] + new_func_lines + lines[end:]

with open('main.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('✅ 替换完成')
print(f'原文件行数：{len(lines)}')
print(f'新文件行数：{len(new_lines)}')
