# 学校服务器 SSH 连接详细教程

## 一、基本信息

| 参数 | 值 |
|------|-----|
| 服务器 IP | `<你的服务器IP>` |
| 用户名 | `<你的用户名>` |
| 端口 | `22`（默认） |

---

## 二、准备工作

### 2.1 什么是 SSH？

SSH（Secure Shell）是一种加密的网络协议，用于安全地远程登录服务器。就像远程桌面，但用命令行操作。

### 2.2 Windows 用户：检查 SSH 是否可用

Windows 10/11 已内置 OpenSSH，打开 **PowerShell** 或 **CMD**，输入：

```bash
ssh -V
```

如果显示版本号（如 `OpenSSH_for_Windows_8.1p1`），说明已安装。

**如果没有安装：**

1. 打开 **设置** → **应用** → **可选功能**
2. 点击 **添加功能**
3. 搜索并安装 **OpenSSH Client**

---

## 三、基本连接方法

### 3.1 直接登录（需要密码）

打开 PowerShell / CMD / 终端，输入：

```bash
ssh <你的用户名>@<你的服务器IP>
```

首次连接会提示：
```
The authenticity of host '<你的服务器IP>' can't be established.
ECDSA key fingerprint is SHA256:xxx...
Are you sure you want to continue connecting (yes/no)?
```

输入 `yes` 并回车，然后输入密码即可登录。

### 3.2 指定端口（如果端口不是默认 22）

```bash
ssh -p 22 <你的用户名>@<你的服务器IP>
```

### 3.3 登录成功后的样子

```
Last login: Wed Apr 9 10:00:00 2025 from 10.1.xx.xx
[你的用户名@master ~] $
```

现在你已经在服务器上了，可以执行 Linux 命令。

---

## 四、配置免密登录（推荐）

每次输入密码很麻烦，配置 SSH 密钥可以实现免密登录。

### 4.1 第一步：生成密钥（在你的本地电脑）

```bash
ssh-keygen -t rsa -b 4096
```

一路回车即可（默认保存在 `C:\Users\你的用户名\.ssh\id_rsa`）：

- `id_rsa` - 私钥（保密，留在本地）
- `id_rsa.pub` - 公钥（需要传到服务器）

### 4.2 第二步：查看公钥内容

Windows PowerShell：
```bash
cat C:\Users\你的用户名\.ssh\id_rsa.pub
```

会看到类似内容：
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC... 用户名@电脑名
```

### 4.3 第三步：将公钥添加到服务器

**方法 A：自动复制（推荐）**

Windows PowerShell（需要 Git Bash 或 OpenSSH 完整版）：
```bash
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh <你的用户名>@<你的服务器IP> "cat >> ~/.ssh/authorized_keys"
```

**方法 B：手动复制**

1. 登录服务器：
   ```bash
   ssh <你的用户名>@<你的服务器IP>
   ```

2. 创建 .ssh 目录（如果不存在）：
   ```bash
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   ```

3. 编辑 authorized_keys 文件：
   ```bash
   nano ~/.ssh/authorized_keys
   ```
   或
   ```bash
   vim ~/.ssh/authorized_keys
   ```

4. 把你本地的公钥内容（`id_rsa.pub` 的全部内容）粘贴进去，保存退出

5. 设置权限：
   ```bash
   chmod 600 ~/.ssh/authorized_keys
   ```

### 4.4 第四步：测试免密登录

退出服务器（输入 `exit`），然后重新连接：

```bash
ssh <你的用户名>@<你的服务器IP>
```

如果不需要输入密码直接登录，配置成功！

---

## 五、进阶：配置 SSH 别名

每次输入 `ssh <你的用户名>@<你的服务器IP>` 比较长，可以设置别名。

### 5.1 编辑本地配置文件

Windows：创建或编辑 `C:\Users\你的用户名\.ssh\config` 文件

添加以下内容：
```
Host school
    HostName <你的服务器IP>
    User <你的用户名>
    Port 22
```

### 5.2 使用别名连接

现在只需要：
```bash
ssh school
```

就可以直接连接了！

---

## 六、常用操作

### 6.1 退出服务器

```bash
exit
```
或按 `Ctrl + D`

### 6.2 从服务器下载文件到本地

```bash
scp <你的用户名>@<你的服务器IP>:/远程文件路径 /本地保存路径
```

例如：
```bash
scp <你的用户名>@<你的服务器IP>:~/data.txt C:\Users\你的用户名\Downloads\
```

### 6.3 从本地上传文件到服务器

```bash
scp /本地文件路径 <你的用户名>@<你的服务器IP>:/远程保存路径
```

例如：
```bash
scp C:\Users\你的用户名\Documents\file.txt <你的用户名>@<你的服务器IP>:~/uploads/
```

### 6.4 传输整个文件夹

下载文件夹（加 `-r` 参数）：
```bash
scp -r <你的用户名>@<你的服务器IP>:~/folder/ C:\Users\你的用户名\Downloads\
```

上传文件夹：
```bash
scp -r C:\Users\你的用户名\Documents\folder <你的用户名>@<你的服务器IP>:~/uploads/
```

---

## 七、常见问题排查

### Q1: 连接超时 / Connection timed out

**原因：**
- 网络不通（可能不在校园网内）
- 服务器防火墙阻止

**解决：**
- 确认你在校园网内，或使用 VPN
- 用 ping 测试：`ping <你的服务器IP>`

### Q2: Permission denied (publickey,password)

**原因：**
- 密码错误
- 公钥未正确添加

**解决：**
- 检查用户名和密码
- 确认 `authorized_keys` 文件权限为 600
- 确认 `.ssh` 目录权限为 700

### Q3: Host key verification failed

**原因：**
- 服务器重装系统后指纹变了

**解决：**
```bash
ssh-keygen -R <你的服务器IP>
```
然后重新连接。

### Q4: Windows 没有 ssh-keygen 命令

**解决：**
安装 Git for Windows，它会附带完整的 OpenSSH 工具。
下载地址：https://git-scm.com/download/win

---

## 八、服务器基本信息

登录后可以查看：

| 信息 | 查看命令 |
|------|---------|
| 系统版本 | `cat /etc/centos-release` |
| 内核版本 | `uname -a` |
| CPU 信息 | `lscpu` |
| 内存信息 | `free -h` |
| 磁盘信息 | `df -h` |
| 当前目录 | `pwd` |
| 用户目录 | `echo $HOME` |

---

## 九、快速参考卡片

```bash
# 连接
ssh <你的用户名>@<你的服务器IP>

# 免密配置
ssh-keygen -t rsa -b 4096
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh <你的用户名>@<你的服务器IP> "cat >> ~/.ssh/authorized_keys"

# 文件传输
scp 本地文件 <你的用户名>@<你的服务器IP>:远程路径
scp <你的用户名>@<你的服务器IP>:远程文件 本地路径

# 退出
exit
```