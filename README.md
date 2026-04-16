# 1 虚拟环境
## 1.1 安装uv包进行包管理
```bash
uv install
```
再创建demouv unit
```bash
uv init demouv
```
## 1.2 创建虚拟环境
进入demouv目录后输入命令
```bash
uv venv
```
## 1.3激活虚拟环境
```bash
.venv\Scripts\activate
```
## 1.4 安装依赖
安装pandas langchain ipykernel依赖
```bash
uv add pandas langchain ipykernel
```

然后就可以再demouv中进行分文件学习
