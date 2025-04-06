# Python基础语法精要（面向Java/JavaScript开发者）

## 1. 缩进与代码块
Python使用缩进来定义代码块，而不是Java/JavaScript中的花括号{}。这是Python最显著的语法特征之一。

```python
# Python的缩进方式
def example_function():
    if True:
        print("这是一个缩进的代码块")
        for i in range(3):
            print(i)  # 更深的缩进

# 对比JavaScript
# function exampleFunction() {
#     if (true) {
#         console.log("这是用花括号的代码块");
#     }
# }
```

## 2. 动态类型系统
Python是动态类型语言，变量不需要声明类型，这一点类似JavaScript但不同于Java。

```python
# Python的动态类型
x = 42          # 整数
x = "Hello"     # 可以直接改为字符串
x = [1, 2, 3]   # 可以改为列表

# Java中需要声明类型：
# int x = 42;
# String str = "Hello";
```

## 3. 列表推导式
这是Python特有的一个强大特性，可以用简洁的语法创建列表，在Java/JavaScript中没有直接对应的语法。

```python
# Python列表推导式
squares = [x**2 for x in range(10) if x % 2 == 0]  # [0, 4, 16, 36, 64]

# JavaScript等价写法：
# const squares = Array.from(Array(10).keys())
#     .filter(x => x % 2 === 0)
#     .map(x => x**2);
```

## 4. with语句和上下文管理
Python的with语句提供了一种优雅的方式来处理资源管理，类似于Java的try-with-resources。

```python
# Python的with语句
with open("file.txt", "r") as file:
    content = file.read()
    # 文件会自动关闭

# Java等价写法：
# try (FileReader file = new FileReader("file.txt")) {
#     // 处理文件
# }
```

## 5. 装饰器
Python的装饰器提供了一种优雅的方式来修改函数或类的行为，类似于Java的注解但更加灵活。

```python
def log_function(func):
    def wrapper():
        print(f"调用函数: {func.__name__}")
        return func()
    return wrapper

@log_function
def hello():
    print("Hello, World!")

# 等价于：
# hello = log_function(hello)
```

## 6. 生成器函数
Python的生成器使用yield关键字，提供了一种内存效率高的方式来处理大量数据。

```python
def count_up_to(n):
    i = 1
    while i <= n:
        yield i
        i += 1

# 使用生成器
for number in count_up_to(3):
    print(number)  # 输出 1, 2, 3

# JavaScript的等价实现使用function*语法：
# function* countUpTo(n) {
#     for(let i = 1; i <= n; i++) {
#         yield i;
#     }
# }
```

## 7. 解包操作
Python提供了强大的解包语法，可以方便地处理序列。

```python
# 列表解包
a, b, *rest = [1, 2, 3, 4, 5]
print(a)     # 1
print(b)     # 2
print(rest)  # [3, 4, 5]

# 在函数调用中解包
def add(x, y, z):
    return x + y + z

numbers = [1, 2, 3]
result = add(*numbers)  # 解包列表作为参数
```

## 8. f-strings（格式化字符串）
Python 3.6+引入的f-strings提供了一种更优雅的字符串格式化方式。

```python
name = "Python"
version = 3.9
message = f"{name} {version} is awesome!"

# 对比JavaScript的模板字符串：
# const message = `${name} ${version} is awesome!`;

# 对比Java：
# String message = String.format("%s %s is awesome!", name, version);
```

## 9. 切片操作
Python的切片操作提供了处理序列的强大方式。

```python
numbers = [0, 1, 2, 3, 4, 5]
print(numbers[1:4])    # [1, 2, 3]
print(numbers[::2])    # [0, 2, 4] - 步长为2
print(numbers[::-1])   # [5, 4, 3, 2, 1, 0] - 反转

# 在Java/JavaScript中需要使用方法组合实现类似功能：
# JavaScript: array.slice(1, 4)
# Java: Arrays.copyOfRange(array, 1, 4)
```

## 10. 字典推导式
类似于列表推导式，Python还提供了字典推导式，这是一个创建字典的强大工具。

```python
# 创建平方数字典
squares = {x: x**2 for x in range(5)}
# 结果: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# JavaScript等价实现：
# const squares = Object.fromEntries(
#     Array.from({length: 5}, (_, i) => [i, i ** 2])
# );
```

这些基础语法特性展示了Python的简洁性和表达力。对于Java和JavaScript开发者来说，理解这些特性不仅有助于学习Python，还能帮助在不同编程范式之间建立联系，提升整体编程能力。

## 11. MySQL数据库连接
Python连接MySQL数据库通常使用`mysql-connector-python`或`pymysql`库，这里展示两种常用方式：

```python
# 方式1：使用 mysql-connector-python
import mysql.connector

# 建立连接
conn = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database"
)

try:
    # 创建游标
    cursor = conn.cursor()
    
    # 执行SQL查询
    cursor.execute("SELECT * FROM users WHERE age > %s", (18,))
    
    # 获取结果
    results = cursor.fetchall()
    for row in results:
        print(row)
        
finally:
    # 关闭连接
    cursor.close()
    conn.close()

# 方式2：使用 pymysql 和上下文管理器
import pymysql
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = pymysql.connect(
        host='localhost',
        user='your_username',
        password='your_password',
        database='your_database',
        charset='utf8mb4'
    )
    try:
        yield conn
    finally:
        conn.close()

# 使用连接
with get_db_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users")
        results = cursor.fetchall()
```

## 12. 多线程编程
Python提供了`threading`模块用于多线程编程，以下是两个常用示例：

```python
# 示例1：基本的线程创建和使用
import threading
import time

def worker(thread_name, delay):
    """线程工作函数"""
    for i in range(3):
        time.sleep(delay)
        print(f"{thread_name}: {time.ctime()}")

# 创建两个线程
thread1 = threading.Thread(target=worker, args=("Thread-1", 1))
thread2 = threading.Thread(target=worker, args=("Thread-2", 2))

# 启动线程
thread1.start()
thread2.start()

# 等待线程完成
thread1.join()
thread2.join()

# 示例2：使用线程池执行并发任务
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def download_url(url):
    """下载指定URL的内容"""
    with urllib.request.urlopen(url) as response:
        return response.read()

# 使用线程池下载多个URL
urls = [
    'http://example.com',
    'http://example.org',
    'http://example.net'
]

with ThreadPoolExecutor(max_workers=3) as executor:
    # 提交任务到线程池
    future_to_url = {executor.submit(download_url, url): url for url in urls}
    
    # 获取结果
    for future in future_to_url:
        url = future_to_url[future]
        try:
            data = future.result()
            print(f"{url}: 下载成功，数据大小 {len(data)} bytes")
        except Exception as e:
            print(f"{url}: 下载失败 - {e}")

# 对比Java线程：
"""
Java线程示例：
public class MyThread extends Thread {
    public void run() {
        // 线程执行的代码
    }
}
// 创建和启动线程
MyThread thread = new MyThread();
thread.start();
"""

# 对比JavaScript异步：
"""
JavaScript使用Promise和async/await：
async function doWork() {
    try {
        const result = await fetch('http://example.com');
        const data = await result.json();
    } catch (error) {
        console.error(error);
    }
}
"""
```

以上展示了Python中处理数据库连接和多线程编程的基本方法。Python的这些特性相比Java和JavaScript都有其独特的语法和使用方式，理解这些差异对于跨语言开发非常有帮助。
