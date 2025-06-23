# MkDocs 指南

欢迎来到我的 MkDocs 学习笔记！

将我所有大学期间学习的所有内容组织成一个包罗万象的笔记仓库，这个想法是在大二春夏学期的最后一门考试结束之时同步萌生出来的。选择使用 MkDocs-Material 作为工具，一定程度上是受本学期专业课——计算机体系结构（Computer Architecture）的实验文档的启发。同时，我在复习各门课程时时常参考往届学长笔记，其中使用 MkDocs 的优秀例子并不少见。

因此，在当天回家的动车上，我以学习 MkDocs 的基本使用方法为期末成绩惨淡的慰藉。这篇文档也自此成为加入我的知识库的第一位成员。

## MkDocs 简介

> [!info] : 什么是 MkDocs？
> - **MkDocs** 是一个**快速、简单、漂亮的静态网站生成器 (Static Site Generator, SSG)**。

MkDocs 的核心使命只有一个：将用**纯文本 Markdown 语言**书写的文档，转换成一个专业、易于导航的**静态网站**。静态网站无需数据库查询和服务器端动态渲染，访问者直接加载最终的 HTML 文件。

## 环境配置

### 编辑器选择

VSCode 是一个轻量级、扩展性好的代码编辑软件，同时具有强大的生态、终端和 Git 集成，有着极佳的一体化体验，是计算机专业学生的好友，也是我的编辑器首选。

我的电脑是 Windows 系统，在多次尝试后，我推荐使用 VSCode + WSL。原因是 MkDocs 是基于 Python 的，在 Windows 上配置 Python 环境有时会遇到路径、编码、依赖库编译等各种奇怪的问题。而 WSL (Windows Subsystem for Linux) 提供了一个完整的、原生的 Linux 环境，使用 `apt` 安装依赖，使用 `pip` 安装 Python 包，完全避开 Windows 的坑。这会让整个开发和部署流程变得极其稳定和可靠。

 > [!hint] 
 > 如果你使用的是 macOS 或者 Linux，可以忽略这段内容。在 VSCode 中配置 WSL 的内容具有一定篇幅，由于和本文主题非强相关，不做过多赘述，可参考这篇文章：[WSL(Windows Subsystem for Linux)的安装与使用](https://www.cnblogs.com/JettTang/p/8186315.html)

### 配置 Python 及其虚拟环境

现在我们已经拥有了一个 Linux 终端，使用命令行或者手动创建一个文件夹作为主工作文件夹。然后进入这个目录，开始依赖包安装的过程。

前文提到， MkDocs 是基于 Python 的，我们需要首先检查当前是否已经安装版本合适的 Python：

```bash
python3 --version
```

如果没有安装，运行如下命令：

```bash
# 首先更新包列表是一个好的习惯
sudo apt update

sudo apt install python3

# 安装完成后检查
python3 --version

# 顺便下载一下 pip（如果失败则向后延至创建虚拟环境之后）
sudo apt install python3-pip
```

在使用 Python 的过程中，使用虚拟环境是常见且推荐的做法，在此仅给出使用 venv 虚拟环境的方法，其他虚拟环境的尝试有待后续补充。

1. 首先确保你在项目文件夹内：

```bash
cd /Your-Directory-Path
```

2. 创建 venv 虚拟环境：

```bash
python3 -m venv venv
```

这可能需要一段时间。然后你会发现在项目根目录下出现了一个 venv 文件夹。

3. 激活虚拟环境：

```bash
source venv/bin/activate
```

此时你的终端命令行首部会出现 `(venv)` 提示，表示你已经处在虚拟环境中。此后所有的依赖下载都应在虚拟环境中，不会与外界互相影响。如果你需要退出虚拟环境，直接运行 `deactivate` 即可。

> [!attention] : 为什么我们需要虚拟环境？
> 这是一个许多在 Linux (包括 WSL) 上使用 Python 的开发者都会遇到的问题。导致这个问题的原因是，Linux 系统为了保护自己而设置的一个安全措施，阻止直接向系统级的 Python 环境中安装新软件包。因此，如果不使用虚拟环境，可能会在 `pip3 install` 时报错。

### 下载依赖

现在我们可以安全地使用 `pip` 来下载我们的主体：mkdocs-material

```bash
pip3 install mkdocs-material
```

此外，Git 的配置对于这个项目来说是必要的，即便你可能没有版本控制的需求，因为 Mkdocs 需要在一个 Github Repository 上执行构建和部署，或者在其他平台上连接 Github Repository。使用命令行确认当前环境是否支持 Git：

```bash
git --version

# 若没有，下载
sudo apt install git
```

> [!hint] 
> Git 的配置与使用同样不再赘述。 


### 可选：使用 Obsidian 等 Markdown 编辑器提升体验

我在尝试 MkDocs 之前惯用 Obsidian 来管理和存储笔记文件。换成 MkDocs 之后，并不意味着我们必须放弃原有的、惯用的 Markdown 编辑器。相反，仍然使用像 Obsidian 这样的编辑器是非常推荐的。具体内容见后文进阶内容。

### 总结

回头来看，这些内容是我们使用 MkDocs 必要的：

- 一个支持 macOS/Linux 终端的代码编辑器（笔者没有尝试直接使用 Windows 终端，有待后续完善）
- python 及其虚拟环境
- Git
- mkdocs-material 依赖包

> [!warning] 
> 你应当妥善考虑 `.gitignore` 文件的内容，针对本文档，以下内容是可以不被同步至 Github 远程仓库的：
> 
> - `.vscode/`
> - `venv/`
> - `.obsidian`（如有）
> - `site/`（MkDocs 生成的站点文件）
> - 其他 Python 中可能出现的无需同步的内容
>    
>    




## 主要功能使用介绍

接下来进入 MkDocs 主要功能的使用介绍。本节聚焦 MkDocs 最基本的操作和功能，快速构建一个 demo，帮助你对 MkDocs 有一个底层了解。

### 创建新的项目

首先确保你处在你期望的根目录（确保当前根目录链接到一个已有的 Github 仓库）：

```bash
cd /Your-Dir-Path
```

使用如下命令在当前文件夹内初始化 MkDocs 项目。

```bash
mkdocs new .
```

注意这里 `.` 就表示在当前目录下。运行这一条命令行后，你会观察到在根目录下出现以下内容：

- `docs/`：这是你的 Markdown 文件存放的地方
	- `index.md`：一个初始化的导览文件
- `mkdocs.yml`：一个全局的配置文件，仅包含网页大标题

### 网页预览

这就是一个简单的 demo，你可以通过如下命令行进行预览：

```bash
mkdocs serve
```

在终端发现预览网址，或者使用 VSCode 自动弹出的窗口访问预览网页。你可以在预览状态下更改你的 Markdown 文件内容，观察终端控制台的信息和预览网页的实时状态。

如果你需要退出预览，在终端中输入 <kbd>ctrl</kbd>+<kbd>C</kbd> 即可。

### 网页发布

假设你已经完成了 Markdown 文件的组织和内容，以及 `mkdocs.yml` 的配置，下一步也就是最后的一步，就是将这个网站发布，从本地走向世界。

最主流，同时也是最便捷的方式，是直接部署到 Github Pages。这是一个由 GitHub 提供的免费静态网站托管服务，整个过程非常简单，核心只有一个命令，但在此之前，我们需要做一些准备和检查工作：

1. 项目根目录与 Github 远程仓库关联
2. 源代码已经推送并提交
3. 虚拟环境已激活

现在我们可以正是进入网页发布的环节了。

- **Step 1**：配置 `mkdocs.yml` 的 `site_url`

你需要明确告诉 MkDocs 您网站的最终线上地址是什么，这样它才能正确地生成所有链接（如CSS、JS文件和页面间的跳转链接）。你需要在 `mkdocs.yml` 文件中加入这一行：

```yaml
# mkdocs.yml
site_url: https://<Your-Github-username>.github.io/<Your-Repository-Name>/
```

> [!warning] 
> - 注意末尾的 `/` 不要忘记
> - YAML 对缩进（两个 <kbd>space</kbd>）特别敏感，请异常小心！

- **Step 2**：执行发布命令

在你的编辑器终端中运行：

```bash
mkdocs gh-deploy
```

执行这条命令后，会自动运行 `mkdocs build`，在项目根目录下创建一个 `site` 文件夹，里面是构建好的完整静态网站（包含所有HTML、CSS、JS文件）。然后，它会在你的本地 Git 仓库中创建一个名为 `gh-pages` 的新分支。它会将 `site` 文件夹中的**所有内容**提交到这个 `gh-pages` 分支。最后，它会将这个本地的 `gh-pages` 分支推送到你在 GitHub 上的远程仓库。

整个过程是全自动的，只需等待命令执行完成，看到类似 `INFO - Pushing 'gh-pages' branch to 'origin'` 的成功信息即可。

- **Step 3**：访问 Github

其实这步你无需进行实质性的操作。你可以打开 Github 的对应仓库，看到我们刚刚 **Step 2** 创建的新分支。然后进入 **Settings-Pages**，就能看到网站部署（这可能需要几分钟的时间）的相关信息，包括网页地址、相关分支、项目根目录等。

自此网页发布的操作全部完成。每次修改内容后，需要将内容推送提交，然后重新运行 `mkdocs gh-deploy`，你的修改才会在网页中反映出来。

## 设计面向你的需求的个性化知识库

上节内容，我们仅仅是创建一个简单的 demo 来初步学习 MkDocs 的使用，本节我们将以笔者需求为例，详细讲讲笔者在构建个性化知识库中所做的工作。

### Markdown 文件的组织与导航设计



## Markdown 扩展功能演示大全

欢迎来到功能演示页面！本页面旨在集中展示 `mkdocs-material` 中各种强大的 Markdown 扩展功能。请对照您的 `mkdocs.yml` 文件，确保相关扩展已开启。（本节大部分内容由 Gemini 2.5 PRO 辅助生成）

旁边的目录（Table of Contents）是由 `toc` 扩展生成的。

### Admonition (标注块)

这是用于高亮信息的最常用功能。

#### 基本类型

> [!note] "这是一个笔记 (Note)"
> 这是一个普通的笔记，用于补充说明。

> [!info] "这是一个信息 (Info)"
> 这是一个通用的信息提示。

> [!todo] "这是一个待办事项 (Todo)"
> - [x] 学习 Admonition 语法
> - [ ] 应用到自己的笔记中

> [!tip] "这是一个技巧 (Tip)"
> `mkdocs serve` 会实时刷新你的网站，非常方便。

> [!question] "这是一个问题 (Question)"
> 如何启用这些功能？答：在 `mkdocs.yml` 中配置 `markdown_extensions`。

> [!warning] "这是一个警告 (Warning)"
> 请注意 YAML 格式对缩进非常敏感。

> [!danger] "这是一个危险操作 (Danger)"
> 永远不要在不理解后果的情况下运行 `rm -rf /`。

> [!example] "这是一个示例 (Example)"
> 这里可以放一些示例性的代码或文字。

> [!bug] "这是一个缺陷报告 (Bug)"
> 在版本 1.0 中，此功能可能存在兼容性问题。

> [!definition]
> aaaa

> [!theorem]
> bbbbbb

> [!lemma]
> cccccc

> [!corollary]
> dddddd

> [!proposition]
> eeeeeee


> [!property]
> This is a property.

> [!axiom]
> This is an axiom.

> [!proof]
> This is a Proof

> [!algorithm]
> This is an algorithm.



#### 可折叠的标注块

> [!example]-
> 将不必默认展示的例子（Example）折叠是一个很好的习惯

> [!example]
> 这是一个嵌套折叠的例子
> > [!note]-
> > 嵌套折叠的内容（note）


---

### 内容格式化与样式

这一部分展示由 `pymdownx` 提供的各种文本样式扩展。

#### 高亮、插入、删除、上下标

* **高亮 (Mark)**: 我们用 ==荧光笔== 来划重点。 `\==荧光笔\==`
* **删除线 (Tilde)**: 这是一个 ~~错误的说法~~。 `~~错误的说法~~`
* **下标 (Tilde)**: H~2~O 是水的化学式。 `H~2~O`
* **插入 (Caret)**: 这是一个 ^^新加入的^^ 文本。 `^^新加入的^^`
* **上标 (Caret)**: a^2^ + b^2^ = c^2^。 `a^2^ + b^2^ = c^2^`

#### 键盘按键样式 (Keys)

在技术文档中清晰地展示快捷键。

* 要复制文本，请使用 <kbd>ctrl</kbd>+<kbd>c</kbd>。
* 要保存文件，请使用 <kbd>cmd</kbd>+<kbd>s</kbd>。
* 组合键如 <kbd>shift</kbd>+<kbd>alt</kbd>+<kbd>f</kbd> 用于格式化代码。

#### 审阅标记 (Critic)

用于记录修改痕迹，在校对自己或与人协作时非常有用。

* **删除**: 这段内容是 {--多余的--}。
* **添加**: 我们需要 {++补充一些新的想法++}。
* **高亮修改**: 这个地方 {==需要重点注意==}。
* **评论**: 我们明天再讨论这个问题 {>>这个截止日期可能太紧了<<}。

---

### 交互式元素

#### 任务列表 (Tasklist)

一个可点击的任务清单。

- [x] 完成第一阶段的学习
- [ ] 开始第二阶段的探索
- [ ] 撰写总结报告

#### 选项卡内容 (Tabbed)

在有限空间内展示多种信息。

=== "Windows"
    ```powershell
    ## 在 PowerShell 中
    echo "Hello, Windows!"
    ```

=== "macOS / Linux"
    ```bash
    ## 在 Bash 或 Zsh 中
    echo "Hello, Unix-like World!"
    ```

=== "包含复杂内容的选项卡"
    !!! info "提示"
        选项卡内可以包含任何其他 Markdown 元素，比如标注块、图片、列表等。

        - 这是一个列表项
        - 这是另一个列表项

---

### 技术与代码相关

#### 数学公式 (Arithmatex)

支持 LaTeX 语法。

* **行内公式**: 著名的质能方程是 $E=mc^2$。
* **块级公式**:
    $$
    \int_{-\infty}^{\infty}
    $$