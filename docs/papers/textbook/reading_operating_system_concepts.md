# Reading: Operating System Concepts

Overview: 

- OS 是计算机用户和计算机硬件之间的媒介
- 目的是给用户提供方便高效地运行程序的环境

- OS 是一个 Software，这个 software 管理计算机硬件，硬件应当提供合适的机制 (mechanism) 保证系统正常运行，并防止程序影响系统

- 操作系统是 piece by piece，因此是 well-delineated (层次分明的)

## Chapter 1: Introduction

- OS 不仅管理硬件，还为 application programs 提供基础

- OS 存在范围极广，尤其在“物联网”设备中

- Computer Hardwares:
	- CPU
	- memory
	- I/O Devices
	- Storage

OS 的另一个职责是将这些硬件资源分配给程序


### Section 1.1: What Operating Systems Do

Four components of the computer systems (from bottom to the top):

1. Hardware
2. Operating System
3. Application Program
4. User

这种视角下，OS 就负责控制 hardware resources，并协调这些在不同用户、不同应用下的使用。

另外一种视角：

1. Hardware
2. Software
3. Data


OS 像政府一样本身不发挥任何有用的功能（你不会为了使用 OS 而打开计算机），而是提供程序能够进行有用的工作的环境。


观察 OS role 的两种 View：User View & System View


- User View

取决于使用的接口 Interface

一个用户独占所有资源，目标是最大化用户正在进行的工作的效率

这样，OS 的设计就需要为了 ease of use，无需关注资源的利用率（即有多少硬件/软件资源被共享）

一些计算机没有 user view，如家庭设备中的 embedded computers（没有 User View 是因为其中的 OS、Applications 被设计为不在用户干预下运行）


- System View

从计算机的角度，OS 是与 hardware 联系最为紧密的 program

看作一个 resource allocator

**分配需要在 efficiently & fairly 之间权衡。**

一种略有不同的视角，更加强调==控制== I/O 和用户程序的需要。OS 是一个用于 ==Control== 的程序，管理用户程序的执行以防止错误的不当操作，尤其关注 I/O 的操作和控制。



OS 之所以存在，正是因为它为解决创建一个可用的计算系统的问题提供了一个合理的方法。

硬件被设计为能够更简单地处理用户程序，但是其本身（裸硬件）不方便使用，所以应用程序出现了。这些程序有着确定的共有的操作，如控制 I/O 设备。OS 则是为了管理这些部分而出现的。

另一种定义：OS 是在计算机上时刻运行的程序——通常称为 kernel (内核)。

其他两种程序：
1. System Program (Associated with OS but ==not necessarily part== of kernel)
2. Application Program

逸闻：微软因为在 OS 中加入过多的 Application functionality 而被起诉，最终被判垄断限制应用程序竞争。


移动设备的 OS 在拥有 kernel 的同时，还包含了 middleware —— 一个用于为应用程序开发商提供服务的 software framework（supports databases, multimedia, and graphics (to name only a few)）。


本书大部分内容集中在 kernel，但为了全面描述 OS，也会涉及其他组件。


### Section 1.2: Computer-System Organization

现代计算机系统：一或多个 CPU，数个设备控制器，通过共同的一个 system bus 连接。

每一个 device controller 负责一个种类的 device

device controller 维护一些本地的 buffer storage，以及一组专用 register，其本身负责他所控制的外部设备和本地 buffer storage 之间的数据传输。

通常对于每一个 device controller，OS 都有一个 device driver。Device driver 理解 controller，并为 OS 提供统一的 interface

如此，CPU 和 devices 就可以并行处理任务，并且在内存上也是共用的（competing for memory cycles）。这就需要一个 memory controller 对访问进行 synchronize 以确保其有序性。

![[fig-1-2.png]]

- Interrupts

直观上，中断 interrupt 提醒 CPU 注意需要关注的事件。

一个例子：driver 将命令加载到 device register 中，controller 读取这些命令，进行一定的操作（数据传输）。当操作完成时，controller 需要告知 driver，这种“告知”的方式就是通过 interrupts 实现的。

通常，硬件可以随时在 system bus 上传输一个信号给 CPU（触发 interrupt）

Interrupts 是计算机系统和硬件交流的重要方式，实际上还有其他多种用途。

Interrupts 一旦发生，CPU 会立刻停止正在做的事，跳转到一个特定的地址，这个地址通常是预设的一段例程，然后开始顺序执行这段例程（回忆《计算机体系结构》实验）。例程执行结束后，CPU 跳转回原先被中断的计算，将其恢复。（这里原文用了 resume 这个词，妙！）

在不同的计算机设计中，interrupts 的机制可能不同，但是几个函数可能是共有的。关键的是，interrupt 必须将控制信号传输到中断服务例程，最直接的做法就是调用一个通用例程来检查中断信息，这个例程再调用中断 handler。

实际上，由于 interrupts 发生的非常频繁，需要尽可能提升处理的速度。采用 pointer table （interrupt vector）的形式间接调用例程是更加高效的，Win & UNIX 都是这样处理的。



从 implementation 的角度，CPU 硬件有一个 interrupt-request line 来感知 interrupt。

几个 terms:
- device controller: *raises* an interrupt
- CPU: *catches* the interrupt & *dispatches* it to handler
- Handler: *clears* the interrupt


上述机制要求 CPU 必须能够处理异步的中断。另外还有一些更加复杂的要求：

1. 需要能够在关键处理时推迟 interrupt 的处理
2. 需要更高效地将 handler 调度给 device
3. 需要能处理多个优先级的 interrupts，并用合适的紧集程度进行响应

大多数 CPU 都有两条 interrupt-request line，分别是 nonmaskable 和 maskable. 对于 maskable interrupt，CPU 可以在执行重要的、不可打断的指令时将其屏蔽，推迟处理。device controller 的 interrupt 大部分是 maskable。