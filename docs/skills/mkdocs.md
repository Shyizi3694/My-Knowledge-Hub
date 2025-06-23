# MkDocs 指南



## Markdown 扩展功能演示大全

欢迎来到功能演示页面！本页面旨在集中展示 `mkdocs-material` 中各种强大的 Markdown 扩展功能。请对照您的 `mkdocs.yml` 文件，确保相关扩展已开启。

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