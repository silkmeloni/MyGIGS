# test_font.py
import matplotlib

# 强制使用无头渲染器（绕过 Linux 缺少图形界面的底层 Bug）
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import shutil

print("1. 正在强行清理 Matplotlib 缓存...")
cache_dir = matplotlib.get_cachedir()
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# 请确保您刚刚通过 ZIP 解压出来的 simhei.ttf 就在当前目录
font_path = './simsun.ttc'

if not os.path.exists(font_path):
    print(f"❌ 找不到字体文件：{font_path}")
else:
    print(f"2. 找到字体文件：{font_path}，文件大小: {os.path.getsize(font_path) / 1024 / 1024:.2f} MB")
    if os.path.getsize(font_path) < 5 * 1024 * 1024:
        print("⚠️ 警告：文件大小异常（小于5MB），字体文件绝对已损坏！请用 ZIP 打包重新上传！")

    try:
        print("3. 正在尝试渲染中文...")
        # 强制加载刚刚解压的字体
        zh_font = fm.FontProperties(fname=font_path, size=16)

        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [1, 4, 9], label='测试曲线')

        # 只要有中文的地方，全部绑定 zh_font
        plt.title('测试中文标题是否乱码 (Test Title)', fontproperties=zh_font)
        plt.xlabel('横轴测试 (X-axis)', fontproperties=zh_font)
        plt.ylabel('纵轴测试 (Y-axis)', fontproperties=zh_font)
        plt.legend(prop=zh_font)

        plt.savefig('test_font_output.png', dpi=300)
        print("✅ 测试成功！请打开当前目录下的 test_font_output.png 查看是否有中文！")
    except Exception as e:
        print(f"❌ 渲染失败，底层环境报错：{e}")