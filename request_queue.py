### request_queue.py
# ----------------------
# 这是全局维护的请求队列，保证推理任务按顺序执行

from queue import Queue

# 创建一个全局可以被多个组件共享的队列
request_queue = Queue()

# Queue 是线程安全的，无需额外锁控
