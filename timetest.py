import time

# 记录开始时间
start_time = time.time()

# 这里是你的代码
# 例如：result = your_function()
time.sleep(2)  # 模拟耗时操作

# 记录结束时间
end_time = time.time()

# 计算并打印耗时
elapsed_time = start_time - end_time
print(f"程序执行耗时: {elapsed_time:.4f} 秒")
print(f"程序执行耗时: {elapsed_time * 1000:.2f} 毫秒")