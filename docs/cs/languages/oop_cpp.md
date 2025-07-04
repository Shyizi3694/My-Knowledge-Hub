# 面向对象程序设计 (C++)


## Chapter 1: Introduction

### Section 1.1: 教学安排
这是 CX 老师班级的 Assessment 分布：

- Lab (50%)
- Final (50%)

这是 CX 老师班级提供的 Slides 的章节结构：

- Chapter 1:   Introduction
- Chapter 2:   Using Objects
- Chapter 3:   Memory Model
- Chapter 4:   Class
- Chapter 5:   Composition & Inheritance
- Chapter 6:   Polymorphism
- Chapter 7:   Design
- Chapter 8:   Copy Constructor
- Chapter 9:   Operator Overloading
- Chapter 10: Streams
- Chapter 11: Templates
- Chapter 12: Iterators
- Chapter 13: Exceptions
- Chapter 14: Smart Pointers
- Chapter 15: Miscellaneous Topic

### Section 1.2: C++ 简介

> [!note] C 语言：PROS & CONS
> - PROS:
> 	- 高效的程序
> 	- 能够直接访问硬件（适用于 OS）
> 	- 灵活性好
> - CONS：
> 	- 类型检查不足
> 	- 不适用于高级应用程序
> 	- 不直接支持面向对象

因此，C++ 的目标就是，将 C 语言的灵活性和高效性与面向对象的支持联合起来。

C 与 C++ 的关系是：

- C++ 建立在 C 之上
- C 的知识有助于你理解 C++
- C++ 支持更多的编程类型
- C++ 提供更多的特征

> [!check]- C++ 的提升之处
> - 数据抽象 Data Abstraction
> - 访问控制 Access Contro
> - 初始化 & 清理 Initialization & Clean up
> - 函数重载 Function Overloading 
> - 输入/输出流 I/O Stream
> - 常量 Constants
> - 名控制 Name Control
> - 内联函数 Inline Functions
> - 引用 References
> - 运算符重载 Operator Overloading
> - 面向对象支持 Support for OOP
> - 模板 Templates
> - 异常处理 Exception Handling
> - 扩展库 Extensive Libraries
> - STL 

这也是《面向对象程序设计（C++）》所学习的大致内容。

尽管 C++ 可以被看作“更好的” C，我们应该像对待一种全新的语言一样对待 C++。

----

## Chapter 2: Using Objects

### Section 2.1: String

> [!info]
> 使用 string 需要注意的点：
> - 必须 `#include <string>`
> - 像这样定义 `string str;`
> - 使用 string 内容进行初始化 `string str = "Hello";`
> - 使用标准输入/出流 `cin >> str; cout << str;`




> [!warning] 
> string 类型变量允许形如 `str1 = str2` 的赋值，但注意字符数组这样赋值是非法的！


- string 类型变量可以通过 `+/+=` 运算进行连接

- string 的构造函数有如下三种写法：
```cpp
string (const char *cp, int len);
string (const string& s2, int pos);
string (const string& s2, int pos, int len);
```

- 获取子串：
```cpp
substr (int pos, int len);
```

- string 类型变量的修改函数：
```cpp
assign ();
insert (int pos, const string& s);
erase ();
append ();
replace(int pos, int len, const string& s);
```

- string 中字段的查找：
```cpp
find (const string& s);
```

### Section 2.2: File I/O

- 注意事先需要导入 `<ifstream>` 和 `<ofstream>`

- 写入：
```cpp
ofsrteam File("filepath");
File << "Hello World" << std::endl;
```

- 读取：
```cpp
ifstream File("filepath");
File >> str;
//print(str);
```

### Section 2.3: STL Introduction


STL（Standard Template Library） 是 C++ 的标准模板库，提供了封装好的数据结构和算法。

> [!important] 为什么我们需要 STL?
> 1. STL 为我们提供了设计好了的工具，减少我们的开发时间
> 2. 增加代码的可读性
> 3. 鲁棒性
> 4. 便携，便于维护


STL 有三个部分：
- Containers 容器
- Algorithms 算法
- Iterators 迭代器

### Section 2.4: STL - Containers

STL 中的容器可以划分为四类：

> [!info]- STL 容器分类
> - Sequential container 序列容器：
> 	- array        (static)
> 	- vector       (dynamic)
> 	- deque        (double-ended queue)
> 	- forward_list (singly-linked)
> 	- list         (doubly-linked)
> - Associative container 关联容器
> 	- set      (collection of unique keys)
> 	- map      (collection of key-value pairs)
> 	- multiset
> 	- multimap
> - Unordered Associate containers 无序关联容器
> 	- unordered_set
> 	- unordered_map
> 	- unordered_multiset
> 	- unordered_multimap
> - Adaptors 适配器
> 	- stack
> 	- queue
> 	- priority_queue


值得注意的是，关联容器的元素也是经过排序的，其内部通过红黑树实现，适用于元素有必要有序存储的场景；无序关联容器的元素未经过排序，为了加速查找、插入和删除，通过哈希表实现，适用于元素无需有序存储的场景，且对性能要求很高。

#### Container - Vector

> [!info]+ Vector 的基本操作
> - Constructor / Destructor
> - Element access
> 	- `at()`
> 	- `operator[]`
> 	- `front()`
> 	- `back()`
> 	- `data()`
> - Iterators
> 	- `begin`
> 	- `end`
> 	- `cbegin`
> 	- `cend`
> - Capacity
> 	- `empty()`
> 	- `size()`
> 	- `reserve()`
> 	- `capacity()`
> - Modifiers
> 	- `clear()`
> 	- `insert()`
> 	- `erase()`
> 	- `push_back()`



#### Container - Map

Map （映射）存储了 *key-value pairs* （键值对），通过键进行查询，并返回值

> [!example]- 电话簿
> - `map<string, string>`


Map 的使用方法如下：

> [!info]
> - Construct:
> - `map<type1, type2> map_name`
> - Access:
> - `var = map_name[key];`
> - Iterators
> - Capacity
> - Modifiers:
> - insert: `map_name[key] = var;`


> [!warning]
> 注意当被访问的键不存在于映射中时，会自动创建该键并将其初始化为默认值（Silent Insertion）。所以在访问时必须先检查键是否已存在：
> - `if (map_name.count("key"){};`
> - `if (map_name.contains("key"){}; // introduced in C++20`


#### Container - Stack (Adaptor)

Adapter （适配器）的设计模式是，适配器实例化你期望的类接口，然后与供应商接口交流来服务于你的需求。

![[oop-img1-adaptor.png]]

Adaptors 不是通过继承，而是将已有的模块（如 Vector）包含在内部，只暴露自身该有的接口。这种方式更加灵活、安全，保证逻辑纯粹性。

### Section 2.5: STL - Algorithms

STL 提供的算法作用于迭代器定义的 $[first,last)$。

> [!info]- 列举一些 STL 的算法函数
> - `for_each()`, `find()`, `count()`,...
> - `copy()`, `fill()`, `transform()`, `replace()`, `rotate()`,...
> - `sort()`, `partial_sort()`, `nth_element()`, ...
> - `set_difference()`, `set_union()`,...
> - `min_element()`, `max_element()`,...
> - `accumulate()`, `partial_sum()`,...




### Section 2.6: STL - Iterators

迭代器 Iterators 将容器和算法连接起来

### Section 2.7: Tips & Pitfalls

#### **Tips:** 
- 使用 `typedef` 来缩短长命名（C++11 后可以使用 `auto` 和 `using`）
- 实现自己的类时可能需要重载赋值操作等
#### **Pitfalls:**
- Access Safety

> [!failure]
> 对于 vector，请避免这样做！
> - `vector_name[idx] = val;`
> 

而是使用 `push_back()` 进行动态扩展，使用构造器进行预分配，使用 `resize()` 进行重分配。

- 充分利用 `empty()` 来代替 `size() == 0` 作为条件语句的判断
- Invalid Iterator

> [!failure]
> 使用 `begin()` 迭代器后（以 `list` 为例），`++itr` 是非法的！

可以充分利用 `erase()` 的返回值：`itr = list_name.erase(itr);`

----

## Chapter 3: Memory Model
本章主要讨论如下几种变量在内存中的存储方式：
- global variables 全局变量
- static global variables 静态全局变量
- local variables 局部变量
- static local variables 静态局部变量
- allocated variables 动态分配内存的变量


存储这些变量的内存是这样规划的：
![[oop-img2-memoryStructure.png]]

从下往上对应低地址到高地址。最上层是用户代码不可见的 Virtual Memory，其次是用户栈 User Stack，由编译器自动管理，向下为 top 指针的方向；向下有一段空闲区域，该区域中间包含共享的库 Shared Libraries。空闲区域的下方是堆，向上增长。再下则是静态/全局存储区，包括 .data 段和 .bss (Block Started by Symbol) 段。最后是代码区。

下面我们讨论上面提到的几种变量在这样的内存结构中是如何存储的。

### Section 3.2: Local Variables

函数参数，局部变量（默认指非静态的）和函数调用的返回地址及其他管理信息定义在函数内，其生命周期与变量所在作用域（通常是函数体（`{...}`）严格绑定。这些内容被==存储在 Stack 区域==中，当变量被创建时入栈，函数执行完毕返回时自动销毁（出栈）。

- 特点：
	- **分配和释放速度极快**：栈是一种后进先出 (LIFO) 的数据结构，内存的分配和回收只是移动一下栈指针 (Stack Pointer)，非常高效。
	- **大小有限**：栈的可用空间通常是固定的，且相对较小（在 Windows/Linux 上通常是几 MB）。如果函数调用嵌套太深，或者局部变量（尤其是大数组）太大，就会耗尽栈空间，导致所谓的“**栈溢出 (Stack Overflow)**”。
	- **线程安全**：每个线程都有自己独立的栈，因此线程之间的局部变量是隔离的，不会互相干扰。

### Section 3.3: Allocated Variables

动态分配内存的变量指通过 `new` 或者 `malloc` 分配内存的变量，其生命周期直到对其进行 `delete` 或 `free` 都不会终止。动态分配内存的变量==存储在 Heap 区域==中，由程序员手动管理。

- 特点：
	- **灵活性高**：可以在程序的任何时候申请任意大小的内存（只要物理内存和虚拟内存足够）。
	- **生命周期长**：对象的生命周期可以跨越多个函数，直到被手动释放。
	- **分配和释放速度较慢**：堆内存的管理比较复杂。操作系统需要找到一块足够大的空闲内存块，并可能需要处理内存碎片问题。这个过程比栈上的分配要慢得多。
	- **产生内存碎片**：频繁的 `new`/`delete` 操作可能导致大量不连续的小块内存散布在堆中，使得之后难以分配大的连续内存块。
	- **内存泄漏风险**：如果忘记 `delete`，这块内存将永远不会被回收，直到程序结束，造成“**内存泄漏 (Memory Leak)**”。

### Section 3.4: Global Variables

全局变量定义在函数之外，可以被多个 .cpp 文件共享。C++ 中有字段 `extern` 与之相关，用于**声明**外部文件或者本地文件的后面内容有一个这样的变量，编译器根据这个声明进行链接，实现全局变量在多文件域中的共享。

> [!warning]
> 注意“定义”（definition）和“声明”（declaration）的区别！！
> - definition 为变量分配内存，而 declaration 不会


全局变量存储在静态/全局存储区中，生命周期从程序开始时创建，程序结束时销毁。已初始化的全局变量存放在 .data 段，初始值从可执行文件中加载；未初始化或初始化为 0 的全局变量存放在 .bss 段，操作系统在程序启动时会将这块区域清零，减少可执行文件的大小。


### Section 3.5: Static Variables

静态变量是指带有 `static` 字段的变量，存储在全局/静态存储区，在不同段内的存储逻辑同全局变量相同。

静态变量的不同类型有着不同的作用：
- 局部静态变量
- 全局静态变量
- 类内静态成员（变量/函数）

> [!info]  局部静态变量和全局静态变量
> - 局部静态变量
> 	- 生命周期：延长至整个程序运行期间
> 	- 作用域：仍然是函数内部
> 	- 用途：需要在多次函数调用中间记住某个值的场景
> - 全局静态变量
> 	- 生命周期：不改变，仍是整个程序运行期间
> 	- 链接属性：内部链接，即仅对当前 .cpp 文件可见，对其他外部文件时隐藏的（不可访问）
> 	- 用途：避免不同文件之间的同名冲突
> - 类内静态变量
> 	- 表示：属于类本身而不是某个特定对象，即类的所有对象共享
> 	- 生命周期：整个程序运行期间，即便一个对象也没有，这个变量仍然存在
> - 类内静态函数
> 	- 调用：可以通过类名直接调用，也可通过对象调用
> 	- 限制：没有 `this` 指针，因而不能访问非静态成员，只能访问其他静态成员
> 	- 用途：用于实现不需要对象实例，与类本身相关的工具函数


### Section 3.6: Pointers to Objects

```cpp
string str = "Hello";
```

这样定义的字符串本身就是一个对象。此时对象被创建并初始化了。

指向对象的指针这样定义：

```cpp
string* ps;
//string* ps = &str;
```

如果是形如第一行的定义，此时定义的确实是一个指向对象的指针，但指针具体指向的内容不清楚。

> [!note] 其他指针操作
> - 获取地址
> `ps = &s;`
> - 获取对象
> `(*ps)`
> - 调用函数
> `ps->length()`


对于动态分配内存的对象，可以直接定义指针接住其返回值来实现指向该对象。

另外，类的非静态成员函数中还存在一个特殊的指针 `this`，指向调用该成员函数的那个对象本身，在需要区分同名成员变量和参数，或者需要从成员函数中返回对象自身的指针/引用时非常有用。

- 可以通过直接 = 赋值。

### Section 3.7: References

Reference 引用是 C++ 中的一种新数据类型，以下是其定义方式和其他形式类似但含义不同的写法的区分：

```cpp
char c; // define a character
char* p = &c; // a pointer to a character
char& r = c; // a reference to a character
```

> [!warning] 关于引用的初始化
> - 对于一般的变量定义，引用是必须显式地初始化的！
> - 对于函数的参数表或者成员变量，引用无需显式地初始化，而是交给调用者或者构造器。


引用与被引用的对象绑定在一起，具有类似指针的功能，但是不分配额外的内存，因此可以看作一种别名或者昵称，在引用上的修改操作会同步反映在被引用的内容上。

> [!important]
> - 引用一经创建，就只能与一个内容绑定，对引用再次赋值不会成为另外一个内容的别名，而是同步将赋值内容拷贝到原被引用内容中。


- 注意引用不能为空，在创建时必须绑定到一个合法的、已存在的对象。这使得它比指针更加安全。

左值指的是可以在 = 左边出现，即可以使用 = 对其进行赋值的内容，一般有命名的变量都是左值。右值指一个临时、即将被销毁的值，通常是字面量或表达式计算结果，无法获取其地址。引用是左值，C++ 中有着严格的规定：

- 一个非 `const` 的左值引用 (`T&`) 不能绑定到一个右值。


> [!failure]
> 因此这样的写法是错误的：
> ```cpp
> void foo(int& a);
> foo(i*3) // error!
> ```




 `const` 字段放宽了这个条件。

- 引用到指针，形如 `int*& a;` 是合法且有用的，本质上是对一个指针的引用

> [!failure]
> - 指针到引用，形如 `int*& a;` 是非法的！！！
> - 引用到引用，形如 `int&& a;` 也是非法的！！！


> [!info] 引用使用的场景
> - 用于修改函数外部的实参
> - 避免对大型对象的昂贵拷贝
> - 在遍历容器时避免对每个元素进行拷贝（如 `for(int& num : numbers) {}`




### Section 3.8: Dynamic Memory Allocation

- `new` & `delete`
- 用法：
	- `new int;`
	- `new object;`
	- `new int[10];`
	- `delete p;`
	- `delete[] p;`

- `new` 和 `delete` 保证了对象的构造和析构的正确调用
- `new` 创建动态数组时，返回块的第一个元素的地址，可以用一个同类型的指针接住，使用 `delete[]` 删除之。

> [!tip]
> - 不要删除同一个块两次
> - 如果使用了 `new`/`new[]`，则必须记得在结束使用时将其 `delete`/`delete[]`
> - 删除空指针是安全的（什么也不会发生）



### Section 3.9: Constants

Constants 常量是对变量的约束和承诺，通过加上 `const` 字段来告诉编译器这个变量是不能够被修改的。

- 带有 `const` 字段的普通变量在定义时必须被正确初始化。

C++ 中常量默认是内部链接，即这个标识符只在当前文件有效，当不同文件都通过引入头文件持有同一个常量时，会各自拥有一个独立副本，而不会出现重定义的错误。可以通过强制加入 `extern` 字段将其变为外部链接（不同文件之间共享）

对于简单的常量，如果编译器发现其值在编译时可以直接确定，进行常量折叠（Constant Folding）而尽可能不为其分配内存。之后像查表一样直接用其值去替换所有用到的地方。也可以通过加入 `extern` 强制分配内存。

#### Compile-Time Constant

编译期常量是指在程序的编译阶段，其值已经确定且不会改变的常量。编译器可以直接将这个常量的值嵌入到生成的代码中，从而提升运行时的效率。  

编译期常量的确定发生在**编译阶段（Compilation Phase）**，而不是预处理阶段或汇编阶段。

- 编译器常量必须显式地初始化


#### Run-Time Constant

运行期常量是在程序运行时确定其值的常量。虽然它在运行过程中保持不变，但其值只有在运行时才能最终确定。

> [!example]
> ```cpp
> int foo() { return 42; }
> 
> //....
> 
> const int a = foo();
> ```


> [!note]
> - 静态数组的大小定义必须是一个编译期常量表达式，而不能是运行期常量表达式！
> ```cpp
> const int size = 12;
> int array[size]; //ok!
> 
> int x;
> cin >> x;
> const int size1 = x;
> //int array1[size1]; // error!!
> ```


使用动态数组如 vector 就没有这样的限制。

#### Pointers with Constants

> [!important] 区分指向常量的指针和指针常量
> - 这是指向常量的指针，指针可以被修改，其解引用不能
> `const int *p = a; // equivalent to "int const *p = a;"`
> - 这是指针常量，其解引用可以被修改，而指针本身不能
> `int * const p = a;`


注意普通指针不能使用常量的解引用来初始化，常量的解引用只能用于指向常量的指针的初始化！！

#### String Literals

String Literals 字符串字面量是像 `"Hello, world!"` 这样的、用双引号括起来的字符序列，其类型本质上是一个常量字符数组，即 `const char[]`，存储在内存中的**只读数据段**，这意味着这块内存区域在程序运行时是**不可修改的**。

下面我们讨论两种关于字符串字面量的声明：

```cpp
char* s = "Hello World!";
```

这是一个指针初始化，指针直接指向内存区域。`s` 是一个普通指针，这原本是不合法的（因为不能用常量解引用来初始化普通指针），但是为了兼容 C，这种写法是被允许的。但是更加安全的写法是给 `s` 加上 `const` 字段。

> [!warning]
> - 此时应该避免对 `s` 的修改，因为其指向的内容是 `const` 的，即只读的。


```cpp
char a[] = "Hello World!";
```

这是一个数组的初始化，首先在只读数据区中创建常量的字符串字面量，然后在栈中创建 `a`，最后将字符串字面量逐个拷贝至栈中。这意味着数组的内容是可以被修改的，因为其内容不再是只读的常量。

### Section 3.10: Tips & Pitfalls

- 你可以将非常量当作常量对待
```cpp
void foo(const int* x);
int a = 15;
foo(&a); // OK!
```

- 你不能将常量当作非常量对待（使用 `const_cast<type>(var)` 可以解除常量状态，但是这样做是非常危险的，可能导致未定义操作（Undefined Behavior））
```cpp
void foo(int* y);
const int c = 20;
// g(&c) // ERROR!!
```

- 你不能修改函数传入的常量参数

> [!info] 常量返回值的讨论
> - 常量在初始化时接住非常量返回值是合法的
> - 非常量在初始化时接住常量返回值也是合法的


- 对于较大的对象，直接拷贝十分昂贵，尽可能使用引用或者指针
- 如果你不想一个值被更改，让它成为常量

---

## Chapter 4: Class

### Section 4.1: Introduction

我们从一个例子开始：

> [!example] 点 Point
> 考虑将一个点（Point）的相关内容封装为一个整体，需要的内容有：
> - 点本身的信息
> 	- 点的位置（二维空间的点则为 x,y）
> - 与点有关的行为
> 	- 打印点的位置
> 	- 移动点至某处（或者将位置加减）


在 C 中，我们使用结构体：

```c
typedef struct point {
	int x;
	int y;
} Point;
void print(const Point* p);
void move(Point* p, int dx, int dy);
```

在 C++ 中，我们可以使用类（Class）进行更进一步的封装：

```cpp
class Point {
public:
	void init(int x, int y);
	void move(int dx, int dy);
	void print() const;
private:
	int x;
	int y;
}
```

上述代码能够很好体现面向对象的思想，我们在下文一一解读。

### Section 4.2: Objects = Attributes + Services

分析我们设计的 Point 内容，有两个部分：
- Data 数据：性质或状态
	- 例子中为点的位置
- Operations 操作：函数
	- 例子中的打印、移动函数

> [!example]- 另一个例子：售票机
> - Datas: 
> 	- price
> 	- balance
> 	- total
> - Operations:
> 	- Show Prompt
> 	- Print Balance
> 	- Insert Money
> 	- Print Ticket
> 
> 用代码表示：
> 
> ```cpp
> class TicketMachine {
> public:
> 	void showPrompt();
> 	void getMoney();
> 	printTicket();
> 	showBalance();
> 	printError();
> private:
> 	const int price;
> 	int balance;
> 	int total;
> };
> ```

> [!note] Object v.s. Class
> - Object 对象
> 	- 代表事物，场景
> 	- 在运行时对信息做出反映
> - Classes 类
> 	- 定义实例的性质
> 	- 与 C++ 其他类型有类似的行为

总的来说，类给出了对象的定义，对象是定义在类上的，是实例化的类。

### Section 4.3: The Coding Paradigm of a Class

在 C++ 中，规范的代码写法是使用分离但同名的一个头文件（`.h`）和一个源文件（`.cpp`）来定义单一的一个类。

类的声明和成员函数的原型需要写在头文件中，所有的函数主体（成员函数的实现）需要写在源文件中，然后 `#include "header.h"`。

这样做的好处是，让头文件作为代码的作者和用户之间的契约，只将头文件中的内容呈现给用户，而隐藏成员函数的具体细节，是一种很好的抽象模式。这种契约由编译器保证强制执行。

### Section 4.4: Building Process

下面讲讲编译器是如何执行这种契约的。

这种契约的执行就是从源代码到可执行文件的过程，即构建流程（Build Process）。

- **Stage 1: 编译（Compilation）**

简单来说，编译器的视野是狭隘的，一次只能看到一个 `.cpp` 文件，称为**编译单元（Compilation Unit）**，并将其转译为机器能看懂的二进制代码，产出一个**目标文件（Object File）**，在 Windows 上为 `.obj` 文件，在 Linux/macOS 上为 `.o` 文件。

在编译一个编译单元时，代码中可能会引用其他源文件实现的内容，`.h` 文件在其中的作用就是向编译器“承诺”，这个内容是真实存在的，具体实现在别处，并且应该严格按照 `.h` 文件中声明的类型或原型使用，让编译器能够照常编译而不会报错。

- **Stage 2: 链接（Linking）**

与编译器不同，链接器（Linker）的视野是**全局**的，接收所有由编译器生成的目标文件（`.obj` 文件），以及你可能用到的标准库或其他第三方库文件（`.lib`, `.a`），每发现一处目标文件中包含引用，在全局查找其真正实现所在的目标文件，将这两部分**链接**起来，最终生成一个单一的、完整的可执行文件 `.exe` 等。

这样来看，一种显然可能发生的问题为**未定义引用（Undefined Reference）**，参考下面这个例子：

> [!example]- 未定义引用例子
> 设想一个类 `A`，其实现源文件 `A.cpp` 中引用了类 `B` 的方法，但是你在编译时只提供了 `A.cpp` 如错误地使用了命令行 `g++ main.cpp A.cpp`，则链接器找不到它所引用的内容所在的目标文件，抛出错误。
> - 正确的做法是，（递归地）给出所有用到的源文件：`g++ main.cpp A.cpp B.cpp`

另一种有可能会发生问题为**重定义（Multiple Definition）**，参考下面这个例子：

> [!example]- 重定义例子：编译阶段
> 设想一个类 `A`，其声明和实现分别放在 `A.h` 和 `A.cpp` 中。另有一个类 `B`，声明和头文件放在 `B.h` 和 `B.cpp` 中，但是 `B.h` 需要 `#include "A.h"`。
> 
> 编译器在编译 `main.cpp` 时，编译器看到 `#include "A.h`，于是将 `A.h` 的内容拷贝到头部，然后看到 `#include "B.h`，然后将 `B.h` 的内容拷贝到头部，包括其中包含的 `#include "A.h`，这就导致了一个编译单元中，`A.h` 出现了两次，抛出错误。

编译阶段的重定义在大多数条件下是不推荐也难以改变项目结构来解决的。其实，我们只需要将头文件写的更加“标准”一些，加入 Include Guard：

```cpp
#ifndef HEADER
#define HEADER

// contents in header.h

#endif
```

这保证了拷贝同一个头文件的内容最多执行一次，从而避免了这种重定义的情况。除此之外，链接阶段也可能发生重定义问题：

> [!example]- 重定义例子：链接阶段
> 假设一个全局的头文件中有一个函数，其实现也包含在其中。这个全局头文件被多个源文件导入，在编译阶段，由于编译器的视野时局限的，不会发生任何问题。但是在链接阶段，两个目标文件引用了两个地址不同但内容相同的文件，链接器判断不了，如果其他地方使用了这个内容，应该使用哪一个副本。

正确的做法是，严格按照“声明和实现分离”的写法。

> [!tip] 
> 在实际工程中，一次参与构建的源文件可能非常多，使用 `g++` 命令行变得非常低效。一些自动化构建工具是非常推荐的：
> - CMake


### Section 4.5: Scope Resolution Operator

`::` 称为**作用域解析运算符（Scope Resolution Operator）**，它的核心功能是用来**明确地告诉编译器，你想要访问的变量、函数或类型到底属于哪一个“作用域 (Scope)”**，从而解决可能出现的命名冲突和歧义。

- **用法 1**：类作用域解析
	`<Class Name>::<Member Name>`
这是 `::` 最常见的用法。它将一个名字（函数、变量、类型等）与一个特定的类或结构体关联起来。

主要应用场景有：

- **1. 在类外定义成员函数**：当你在类的声明（通常在 `.h` 文件中）中只声明了函数原型，而在类定义的外部（通常在 `.cpp` 文件中）提供其实现时，你必须使用 `ClassName::` 来告诉编译器这个函数属于哪个类。

```cpp
// MyClass.h
class MyClass {
    void my_func();
};

// MyClass.cpp
void MyClass::my_func() { // 必须用 MyClass:: 来指明作用域
    // ... implementation ...
}
```

- **2. 访问静态成员**：由于静态成员属于类本身，而不是某个特定对象，我们通常通过类名来访问它们。

```cpp
MyClass::static_variable = 100;
MyClass::static_function();
```

- **3. 消除继承中的歧义**：当派生类覆盖了基类的同名函数时，如果你想在派生类内部明确地调用基类的版本，就需要使用 `BaseClassName::`。

```cpp
void Derived::some_func() {
    // ...
    Base::some_func(); // 明确调用基类的版本
    // ...
}
```

- **用法 2**：全局作用域解析（`::member`）

当 `::` 运算符的**左边没有任何东西**时，它代表**全局命名空间 (global namespace)**，也就是你的程序中最外层的作用域。它的主要作用是，当一个**局部变量**或**类成员**与一个**全局变量**或**全局函数**同名时，用来明确地访问那个**全局**的版本。

### Section 4.6: PImpl Technique ()

上面内容是“声明和实现分离”的标准写法，其目的是只将头文件中的内容呈现给用户，而隐藏成员函数的具体细节，是一种很好的抽象模式。但是这种写法的局限性在于，必须在头文件中给出私有成员变量的声明，这一定程度上降低了隐藏性，同时，这意味着，任何包含该头文件的代码，都会在编译时依赖于该类的所有私有成员类型的定义。

为了解决这种方法，产生了“指针到实现”（Pointer to Implementation）的技术。这种技术通过将所有私有成员进一步封装为一个类，只剩下该类的声明和一个指向该类的指针，以此进一步隐藏私有成员的细节。通过 PImple Technique，你的头文件会变成这样：

```cpp
// MyClass.h
#include <memory> // for std::unique_ptr

class MyClass {
public:
    MyClass();
    ~MyClass(); // 析构函数必须在 .cpp 中实现
    void do_something();
private:
    // 用户完全看不到 Impl 里面有什么
    class Impl; // 只需要向前声明
    std::unique_ptr<Impl> pimpl; // 只有一个指向实现的指针
};
```

**降低编译依赖（Remove Compilation Dependency）** 是 PImpl Tech 是最主要、最强大的动机。

- **标准做法的问题**：如果你的 `MyClass.h` 的 `private` 部分有一个 `SomeLibrary::Widget m_widget;` 成员，那么 `MyClass.h` 就必须 `#include <SomeLibrary/Widget.h>`。现在，任何 `#include "MyClass.h"` 的文件，都会间接地依赖于 `SomeLibrary/Widget.h`。如果 `Widget.h` 的开发者修改了他们的文件，那么**所有**包含了 `MyClass.h` 的文件都必须**重新编译**，即使它们根本不关心 `Widget`。在大型项目中，这会导致漫长的编译时间，被称为“编译依赖地狱”。
- **Pimpl 的做法**：
    1. `MyClass.h` 中不再有 `m_widget` 成员，自然也就不需要 `#include <SomeLibrary/Widget.h>`。
    2. 真正的 `m_widget` 成员被移到了只在 `MyClass.cpp` 中定义的 `MyClass::Impl` 结构体里。
    3. 只有 `MyClass.cpp` 这**一个文件**需要 `#include <SomeLibrary/Widget.h>`。
- **结果**：现在如果 `Widget.h` 被修改，**只需要重新编译 `MyClass.cpp` 这一个文件**！所有其他只是用到了 `MyClass` 接口的文件都安然无恙，无需重新编译。这极大地**降低了模块间的编译耦合**，显著缩短了大型项目的构建时间。

Pimpl 的使用存在争议，它在带来好处的同时，也引入了新的成本，所以是否使用它是一个需要权衡的“争议点”。

- **运行时开销**：
    1. **动态内存分配**：需要在构造函数中 `new` 一个实现对象，这有性能开销。
    2. **指针间接寻址**：所有对成员的访问都需要通过一次指针的间接寻址 (`pimpl->...`)，这比直接访问成员变量要慢一点点。对于性能极其敏感的热点代码，这可能是个问题。
- **代码复杂性**： 它增加了代码量和实现的复杂度。你需要维护一个额外的 `Impl` 类，并且主类中的所有方法都需要通过 `pimpl` 指针进行“转发调用”。

### Section 4.7: Oject-oriented Programming with Classes

自此我们可以稍微总结一下目前我们所了解的“面向对象编程”的特征和含义：

> [!info] 
> - 所有东西都是对象
> - 一个程序是一组对象组成的，它们彼此之间通过==发送信息==（Messages）告诉对方应该做什么
> - 每一个对象拥有自己的内存空间，这也是由其他对象组成的
> - 任一对象都有其类型
> - 某一特殊类型的所有对象能够接收相同信息

这里的**信息**是什么呢？信息是怎么交互的？

信息由发出方对象组成，由接收方对象转译。接收方对象在收到信息之后，可能返回一个结果，也可能导致接收方对象的状态变化。信息的交互是由方法调用传递的。

此外，类的使用还体现了封装（Encapsulation）和抽象（Abstraction）的思想。

> [!info] 封装
> - 封装将数据和方法捆绑在一起
> - 隐藏了内部处理数据的细节
> - 限制了访问权限，用户只能访问到 public 的方法

> [!info] 抽象
> - 抽象是指忽略部分细节，专注于上层问题的能力
> - 模块化（Modularization）是将整体分为部分的过程，并且使得部分之间能够独立构建，良好交互。

### Section 4.8: Constructors and Default Constructor

我们可以手动地给一个类















上述 Point 例子无法保证初始化动作一定发生，依赖于程序员的自觉，这是我们不愿看到的。解决办法是是使用参数化构造器 ctor (constructor)。
```cpp
class Example
{
private:
	int x, y;
public:
     Example(int a, int b); // Parameterized constructor
     //....
};
```

这样就要求 Example 在被调用时必须要进行初始化，否则在编译时会出错。若不加入构造器，则编译器会自动加入缺省构造器 dtor (default constructor)


# 24th_Mar: constructor

- When to Ctor/Dtor ?
	- 全局变量：`main()`之前创建 constructor，所有程序结束后析构
	- 函数内部的静态局部变量：第一次调用运行至此处时创建 constructor
	- 函数内部的本地变量：每次运行至此处时创建 constructor
	- `new`时创建 constructor，`delete`时析构

- RAII：
	在函数内部进行`lock()`（构造函数），函数以任意方式退出时`unlock()`（析构函数）。

- 隐藏的成员函数：`this`

- Member Function 可加 `const` 修饰，即:
```cpp
void foo() const {} //read only
```
加入了 `const` 之后只有设置为只读，才能通过编译。


- Static Members

```cpp
struct X {
	static void f(); //declaration
	static int n;//declaration
};
int X::n = 0;//definition
void X::f()
{
	n = 1;
}//definition
```
加了`static`的变量不算完成定义，如果没有上述`//definition`的部分则无法通过编译。
```cpp
struct A{
	static int data;
	A(){}
	void setdata(int i){data = i;}
	void print(){}
};
int main()
{
	A a,b;
	a.setdata(20);
	a.print();
	b.print();
}
```
此处变量`data`不存储在结构体内部，而是在结构体外部（全局变量区），所以`print`结果为两个20。但不能在`static int data; A(){data = 0};` 的情况下，在使用`print()`之前不使用`setdata()`（还未定义）。

- Inline Function: 将调用部分直接在调用处在线展开（类似于`# define `，但不是简单的文本替换，可以避免冗余的自增、函数调用的操作），减少函数调用产生的代价。
```cpp
inline int f(int i)
{
	return i*2;
}
int main()
{
	int a = 4;
	int b = f(a);
}
//------>
int main()
{
	int a = 4;
	int b = a * 2;
}
```
- `inline`有允许重定义的机制。

----
# 24th_Mar: Composition & Inheritance
（组合和继承）

- Composition
	- Direct, Own
	- Reference, share
- Inheritance
	- Base Class & Derived Class

```cpp
struct A
{
	int x,y;
};
struct B
{
	A a;
};
struct C : public A //继承类 A
{
};
int main()
{
	B b;
	B.a.x;//B 需要通过 A 来访问 其中的变量
	
	C c;
	C.x;// C 中包含所有 A 有的，可以直接访问
}
```
从二进制层面上，`b`和`c` 没有差别，只在设计层面不同。

数据在基类中是 `private`时，在派生类中一样不能直接访问，但可以通过成员函数访问。

如果在基类中对成员函数加了`protected`限制，则在外界(`main()`)中不能访问该成员函数，但可以利用派生类中的方法调用该函数，从而间接访问该成员函数。一般来说把数据全部做成`private`而非`protected`。


# 31st_Mar: 继承

- 假设Manager继承Employee，并加入了属于自己的字段title，则构造函数初始化需要==同时显式地使用基类==（Employee）进行初始化。

```cpp
Manager() : title("str"), Employee(....){}
```


- 带继承的构造和析构的顺序：
	先调用派生类的初始化，故基类的构造会先被调用，然后是派生类的构造，派生类的析构，基类的析构

- 派生类中调用了其他的类：
	其他类的构造==先于派生类==的构造而==后于基类==的构造。

- 派生类的方法引用基类方法：

```cpp
void Manager::funct(){
	Employee::funct();
}
```

若如上定义，则原先 `Employee::funct` （同名函数）不能直接引用（name hiding）。

- `friend`：授权访问
	- 数据
	- 函数
	- 类

```cpp
friend void funct(   );
friend int data;
friend struct S;
friend class C
```

- 可以通过指向类的指针对类中的 private members 进行访问（只需知道类中各个成员在内存中的二进制布局）！！

- 在继承时也可以使用`public, private, protected`关键词


|                  | public     in A  | protected in A    | public in A  |
| ---------------- | ---------------- | ----------------- | ------------ |
| B: private     A | private     in B | private      in B | public  in B |
| B: protected A   | protected in B   | protected  in B   | public  in B |
| B: public      A | public      in B | protected  in B   | public  in B |


???校对上表


- Upcasting (向上造型)
```cpp
Manager pete(....);
Employee* ep = & pete;
Employee& er = pete;


ep->print();
```



# 多态


- 存在派生类和多态的情况下，基类的析构需要设置成虚函数并且设置成实体。

- 虚函数 Virtual Functions
- 动态绑定 Dynamic Binding
- 纯虚函数 Pure Virtual Binding

```cpp
#include <iostream>
using namespace std;

// 基类 Animal
class Animal {
public:
    // 虚函数 sound，为不同的动物发声提供接口
    virtual void sound() const {
        cout << "Animal makes a sound" << endl;
    }
    
    // 虚析构函数确保子类对象被正确析构
    virtual ~Animal() { 
        cout << "Animal destroyed" << endl; 
    }
};

// 派生类 Dog，继承自 Animal
class Dog : public Animal {
public:
    // 重写 sound 方法
    void sound() const override {
        cout << "Dog barks" << endl;
    }
    
    ~Dog() {
        cout << "Dog destroyed" << endl;
    }
};

// 派生类 Cat，继承自 Animal
class Cat : public Animal {
public:
    // 重写 sound 方法
    void sound() const override {
        cout << "Cat meows" << endl;
    }
    
    ~Cat() {
        cout << "Cat destroyed" << endl;
    }
};

// 测试多态
int main() {
    Animal* animalPtr;  // 基类指针

    // 创建 Dog 对象，并指向 Animal 指针
    animalPtr = new Dog();
    animalPtr->sound();  // 调用 Dog 的 sound 方法
    delete animalPtr;    // 释放内存，调用 Dog 和 Animal 的析构函数

    // 创建 Cat 对象，并指向 Animal 指针
    animalPtr = new Cat();
    animalPtr->sound();  // 调用 Cat 的 sound 方法
    delete animalPtr;    // 释放内存，调用 Cat 和 Animal 的析构函数

    return 0;
}
```


- 纯虚函数要求子类必须对父类的该函数进行重写
- 纯虚函数使类变为抽象类，无法实例化，只能通过继承之的子类进行实例化
```cpp
#include <iostream>
using namespace std;
 
class Shape {
public:
    virtual int area() = 0;  // 纯虚函数，强制子类实现此方法
};
 
class Rectangle : public Shape {
private:
    int width, height;
public:
    Rectangle(int w, int h) : width(w), height(h) { }
    
    int area() override {  // 实现纯虚函数
        return width * height;
    }
};
 
int main() {
    Shape *shape = new Rectangle(10, 5);
    cout << "Rectangle Area: " << shape->area() << endl;  // 输出: Rectangle Area: 50
    delete shape;
}
```




# Design


## Root Finding Algorithm: Newton's Method

$$
x = x - \frac{f(x)}{f'(x)}
$$


- 很快，二次收敛

- Goal: 实现一个牛顿法计算器

>Example: 求 $\sqrt{2}$ 的值
>
>Solution: 构造 $x^2 = 2$，用牛顿法求根

- 循环跳出条件 （控制精度和最大迭代次数）：
```cpp
while(fabs(x * x - 2) > 1e-12 && (k++) < 1 ) {
	// iterations
}
```

- 更加抽象：求 $\sqrt{a}$ ，同理

- 再抽象
	- 加入 `tolerance` 和 `max_iter`
- 再抽象：封装
```cpp
class NewtonSolver {
private:
	double a;
	double tolerance;
	int max_iter;

	int k;
	double x;
public: 
	NewtonSolver(.....) : ....... {}
	void print_info(....){.....}
	double f(double x) {.....}
	double df(double x) {...}
	bool is_close(double x) {....} //零点接近程度判断，作为循环跳出的条件
 	void improve(double x0) {...} // 传入初值，并进行迭代
}
```

- 再抽象：函数 `f(), df(), a` 是另一个问题，可以拆解出 `NewtonSolver`

将 `f(), df()` 做成纯虚函数，强制用户进行实现


```cpp
//class NewtonSolver {
//
private:
	virtual double f(double x) = 0;
	virtual double df(double x) = 0;

//}

// class SqrtSolver : public NewtonSolver 
//

double f(double x) override
{
//
}
double df(double x) override
{
//

}

//}
```


- 函数式编程：

```cpp
# include <functional>

using fn = std::function<double(double)>;

fn f;
fn df;
```


# Copy Constructor



- Signature: 
	```cpp
	T::T(const T&)
```

- 拷贝类中的指针成员，会导致复制的指针与原指针指向同一内容（只复制了地址，但是实际想要的是复制的指针能够指向另一个内容）
- 两个对象管理同一个内存时（使用new），在析构的时候可能导致同一块内存被delete了两次（由编译器生成默认的复制构造器）。
- 当有指针字段的时候，复制构造器就需要自行编写
	对上述问题的一种解决方式是在复制构造器中管理一块新的内存，将原内存中内容拷贝到新的内存块中，这样在复制后两个对象管理不同的内存，避免了上述问题


- 对象切割：
	假设类 B 继承自 A，A 中有一个拷贝构造器，
```cpp
B b(3,4);
A r2 = b;
```
其中 `r2` 是纯粹的 A 类对象，在从对象 `b` 拷贝属于 A 的部分时，切割掉了 B 的部分，而不会使 `r2` 变成 B 类对象。



- 返回值优化
```cpp
Person bar(const char* s)
{
	return Person(s);
}
```

在调用以上函数（传入一个字符串）对另一个 Person 对象初始化时，表面上看发生了 Copy Constructor 的调用（创建了一个临时的 Person 对象，用临时对象对另一个对象作拷贝），其实在编译的时候这个步骤被优化掉了。

# 21st_Apr

- operator 的重载





-----

# 自定义类型



----





- Default Argument
```cpp
int harpo(int n, int m = 5, int j = 5);

beeps = harpo(2); // n = 2, m = 5, j = 5
beeps = harpo(1,8); // n = 1, m = 8, j = 5
beeps = harpo(1,8,6); // n = 1, m = 8, j = 6
```
- 默认参数必须从右往左加入

### Function Overloading

- 参数类型的重载
```cpp
void foo(int n);
void foo(double n);

foo(2);
foo(2.3);
```


###  Templates
- 相同逻辑，不同元素类型
- 避免重复的代码
- 避免使用公共的基类
- 解决 Untyped List 不安全的问题

- Function Template
- Class Template
	- Containers: stack, list, queue,...
	- Template Member Function


#### Function Templates

```cpp
template <class T>
T foo(T n)
{
	T a;
	return n + a; 
}
```

- T 可以出现在各种位置
- 实例化（发生在调用函数时）后，生成对应类型的函数版本

```cpp
void print(T a, T b)
{
///
}

// print(3, 2.5) 错误表达，两个参数类型不同，编译器不知道要怎么生成
print<double>(2, 5.3) //正确
```

- 当同时具备模板和具体类型的普通函数，调用时先寻找是否有普通函数

#### Class Template





# Iterator

- 无需知道容器内部的数据结构，对容器进行顺序访问
- 解除容器和算法的耦合，形成一种通用的算法

