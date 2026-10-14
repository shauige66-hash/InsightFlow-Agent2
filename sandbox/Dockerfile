# P2-A 沙箱镜像：Python 分析与 Matplotlib 绘图的容器执行环境。
#
# 构建与运行分离：镜像在构建期联网安装依赖，运行期一律 --network none，
# 因此 docker run 不会拉取任何东西。构建脚本见 sandbox/build_image.py。
#
# 为什么需要 fontconfig 与 CJK 字体：
#   - python:3.11-slim 不含任何系统字体（/usr/share/fonts 不存在）；
#   - matplotlib 在 Linux 上通过 fontconfig 发现系统字体，只装字体文件而没有
#     fontconfig 是找不到的；
#   - 项目图表标题/图例含中文，只有 DejaVu（matplotlib 自带）时会渲染成方框。
# 注意不要设置 MPL_IGNORE_SYSTEM_FONTS：它会让 matplotlib 完全忽略系统字体。

# 基础镜像按 **digest** 引用，而不是可变 tag。
#
# 为什么不用 `python:3.11-slim`：tag 会被上游重推，同一个 tag 在不同时间
# 可以指向不同内容，"锁定"就失去意义。
#
# 本 digest 的核验情况（见 P2-A 交付报告 D 节）：
#   - 本地 inspect：Os/Arch = linux/amd64，Id = 该 digest，容器内 Python 3.11.17；
#   - 用 `python@sha256:...` 直接 run 成功并报告 machine=x86_64、system=Linux。
#
# **遗留边界**：本机 registry 不可达（registry-1.docker.io HTTP 000），
# 因此无法取得官方的 manifest-list digest 与之比对。这里记录的是本机已核验的
# linux/amd64 镜像 digest；换机器构建时若 registry 的 manifest digest 与之不同，
# 构建会**明确失败**（拉取不到），不会静默换用别的镜像。
FROM python@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce

# 系统包：fontconfig 供 matplotlib 发现字体，fonts-noto-cjk 提供中文字形。
# 只装必要组件，不装推荐包以控制体积。
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      fontconfig \
      fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*

# 精确锁定的 Python 依赖
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-compile -r /tmp/requirements.lock \
 && rm -f /tmp/requirements.lock

# 容器执行入口（由宿主只读挂载，模型代码不可写）
WORKDIR /sandbox

# matplotlib 与 fontconfig 都需要可写缓存目录；根文件系统是只读的，
# 因此二者都指向 tmpfs。
ENV MPLCONFIGDIR=/tmp/mpl \
    XDG_CACHE_HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

# 默认以非 root 运行；docker run 仍会显式传 --user，这里是第二道保证
USER 65534:65534
