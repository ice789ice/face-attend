import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from fpdf import FPDF, XPos, YPos # 使用 fpdf2 库
import matplotlib.font_manager as fm
import sys # 用于检查操作系统以处理字体路径
import traceback # 用于打印详细错误信息

# ================= 路径配置 =================
class Paths:
    # !!!请根据您的实际情况修改以下路径!!!

    # REPORT_FILE: 您要处理的 Excel 报告文件的完整路径
    REPORT_FILE = r"D:\face-attend\WorkAttendanceSystem-master\V2.0\detection_report.xlsx"

    # DETECTION_RESULTS_DIR: 带有检测框的图片所在的目录。
    # 这是第一个脚本保存检测结果图片 (.jpg) 的地方。用于样本图片抽取。
    # 如果您的第一个脚本保存在其他位置，请修改这里。
    DETECTION_RESULTS_DIR = r"D:\face-attend\WorkAttendanceSystem-master\对比测试模板包\results"


    # ANALYSIS_DIR: 所有分析结果（统计文件、图表、PDF报告、样本、可疑结果）存放的主目录
    # 根据您的要求，设置为 D:\face-attend\WorkAttendanceSystem-master\V2.0\analysis
    ANALYSIS_DIR = r"D:\face-attend\WorkAttendanceSystem-master\V2.0\analysis"

    # 子目录 (这些将在 ANALYSIS_DIR 下创建)
    SAMPLE_CHECK_DIR = os.path.join(ANALYSIS_DIR, "sample_check") # 抽样查看的图片
    SUSPECT_DIR = os.path.join(ANALYSIS_DIR, "suspect_results") # 可疑结果列表
    VIS_DIR = os.path.join(ANALYSIS_DIR, "visualizations") # 图表


# ================= 中文字体设置 =================
def get_chinese_font():
    """尝试查找系统中可用的中文字体路径"""
    # 常用 Windows 字体路径
    possible_fonts_win = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
         "C:/Windows/Fonts/simfang.ttf", # 仿宋
         "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑 bold
    ]

    # 尝试在 Windows 下查找
    if sys.platform.startswith('win'):
        for font in possible_fonts_win:
            if os.path.exists(font):
                print(f"找到字体: {font}")
                return font

    # 尝试使用 matplotlib 的 font_manager 查找
    try:
        # 优先查找常用中文族字体
        # 注意：'Arial Unicode MS' 在一些系统上可能不是默认安装
        font_paths = fm.findfont(fm.FontProperties(family=['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Zen Hei']))
        if isinstance(font_paths, list): # findfont might return a list
             font_path = font_paths[0]
        else:
             font_path = font_paths
        if os.path.exists(font_path):
             print(f"通过 font_manager 找到字体: {font_path}")
             return font_path
    except Exception as e:
        print(f"font_manager 查找字体失败: {e}")

    # 如果所有方法都找不到字体，则抛出错误
    raise FileNotFoundError("未找到可用的中文字体。请确保您的系统安装了中文字体（如 SimHei, SimSun, Microsoft YaHei 等），或手动修改 get_chinese_font 函数指定字体路径。")

# ================= PDF报告类 =================
class PDFReport(FPDF):
    """自定义FPDF类，用于生成带中文的报告"""
    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        # 为 FPDF 添加字体，移除已弃用的 uni=True 参数
        self.add_font("Chinese", "", font_path)
        self.set_font("Chinese", "", 12) # 设置默认字体
        self.set_margins(15, 15, 15) # 设置页面边距 (左, 上, 右)
        self.set_auto_page_break(True, margin=15) # 自动分页，底部边距15

    def header(self):
        """页面头部"""
        self.set_font("Chinese", "", 16) # 设置头部字体
        # cell(宽度, 高度, 文本, 边框, 换行, 对齐方式)
        self.cell(0, 10, "人脸识别分析报告", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(10) # 换行
        self.set_font("Chinese", "", 12) # 恢复默认字体

    def footer(self):
        """页面底部"""
        self.set_y(-15) # 距离底部15mm
        self.set_font("Chinese", "", 9) # 设置底部字体
        # {nb} 会被 FPDF 替换为总页数 (需要 finish() 后才准确)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")
        self.set_font("Chinese", "", 12) # 恢复默认字体


# ================= 分析函数 =================
def setup_dirs():
    """创建存放分析结果的目录"""
    print(f"创建分析目录树于: {Paths.ANALYSIS_DIR}")
    # 确保 ANALYSIS_DIR 本身存在，再创建其子目录
    os.makedirs(Paths.ANALYSIS_DIR, exist_ok=True)
    os.makedirs(Paths.SAMPLE_CHECK_DIR, exist_ok=True)
    os.makedirs(Paths.SUSPECT_DIR, exist_ok=True)
    os.makedirs(Paths.VIS_DIR, exist_ok=True)
    print("分析目录结构创建完成。")

def load_data():
    """加载检测报告文件 (.xlsx 或 .csv)"""
    print(f"尝试加载报告文件: {Paths.REPORT_FILE}")
    if not os.path.exists(Paths.REPORT_FILE):
         raise FileNotFoundError(f"未找到检测报告文件：{Paths.REPORT_FILE}")

    try:
         # *** 修改这里以适应新的Excel格式（有表头）***
         # 读取Excel文件，带表头 (header=0, 默认就是0)
         df = pd.read_excel(Paths.REPORT_FILE, header=0)

         print(f"成功加载 Excel 文件。文件共 {len(df)} 行，{len(df.columns)} 列。")
         # 打印前几行数据和列名，供调试参考
         print("\n加载文件前5行示例:")
         print(df.head().to_string())
         print("\n加载文件列名:")
         print(df.columns.tolist())
         print("--------------------------\n")

         return df
    except Exception as e:
         raise IOError(f"加载 Excel 文件 {Paths.REPORT_FILE} 失败: {e}")


def generate_stats(df):
    """从 DataFrame 中计算并生成统计数据"""
    print("正在计算统计数据...")
    # !!! 根据 load_data 中读取到的表头列名引用 !!!
    face_count_col = "人脸数" # 根据最新的Excel截图表头修改
    status_col = "状态"     # 根据最新的Excel截图表头修改
    # *** 根据最新的Excel截图，成功的标准字符串是 "成功" ***
    success_status = "成功" # 成功的标准字符串

    # 确保列名存在
    if face_count_col not in df.columns or status_col not in df.columns:
         raise ValueError(f"加载的文件缺少预期的列名。请检查 Excel 文件表头是否包含 '{face_count_col}' 和 '{status_col}'，或修改代码中的列名。")

    total_images = len(df)
    # 确保人脸数量列是数字类型，否则转换为数字，无法转换的设为0
    df[face_count_col] = pd.to_numeric(df[face_count_col], errors='coerce').fillna(0).astype(int)

    # --- 检查和清理 '状态' 列 ---
    # print("\n--- 检查和清理 '状态' 列内容 ---") # 可以选择打印或不打印这些详细信息
    if status_col in df.columns:
        # 将状态列转换为字符串类型，防止出现非字符串类型的值
        df[status_col] = df[status_col].astype(str)
        unique_statuses = df[status_col].unique()
        print(f"报告中 '{status_col}' 列的唯一原始值: {unique_statuses}")
        # 尝试清理字符串（去除首尾空格和换行符）
        df[status_col + '_cleaned'] = df[status_col].str.strip().str.replace('\r', '').str.replace('\n', '')
        cleaned_statuses = df[status_col + '_cleaned'].unique()
        print(f"报告中 '{status_col}' 列清理后的唯一值: {cleaned_statuses}")

        # 检查清理后的值是否匹配成功标准
        if success_status in cleaned_statuses:
            print(f"清理后的状态中找到了成功的标准字符串: '{success_status}'")
        else:
             print(f"[警告] 清理后的状态中未找到成功的标准字符串: '{success_status}'。请检查原始数据是否正确。")

        # Use the cleaned column for comparison
        status_col_for_comparison = status_col + '_cleaned'
    else:
        # If status column is missing
         print(f"[警告] 报告文件中未找到 '{status_col}' 列，无法进行状态统计。")
         status_col_for_comparison = status_col # Use missing column name, comparison will yield 0 matches


    # --- Debugging Status Comparison ---
    print(f"\n--- Debugging Status Comparison ---")
    print(f"Comparing column '{status_col_for_comparison}' with success status '{success_status}'")
    count_matching_success_status = 0
    count_not_matching_success_status = total_images # Default to all as non-matching

    if status_col_for_comparison in df.columns:
        # Count how many rows match the success status
        rows_matching_success_status = df[df[status_col_for_comparison] == success_status]
        count_matching_success_status = len(rows_matching_success_status)
        print(f"Number of rows exactly matching '{success_status}' (after cleaning): {count_matching_success_status}")

        # Count how many rows do NOT match the success status
        rows_not_matching_success_status = df[df[status_col_for_comparison] != success_status]
        count_not_matching_success_status = len(rows_not_matching_success_status)
        print(f"Number of rows NOT matching '{success_status}' (after cleaning): {count_not_matching_success_status}")

        # Check original column if cleaning was done
        if status_col in df.columns and status_col != status_col_for_comparison:
             rows_matching_original_success = df[df[status_col].astype(str) == success_status]
             count_matching_original_success = len(rows_matching_original_success)
             print(f"Number of rows exactly matching original '{success_status}' (without cleaning): {count_matching_original_success}")


    print(f"---------------------------------\n")
    # --- End Debugging ---


    success_count = count_matching_success_status
    detection_failures = count_not_matching_success_status


    total_faces_detected = int(df[face_count_col].sum())
    average_faces_per_image = round(df[face_count_col].mean(), 2)
    images_with_zero_faces = len(df[df[face_count_col] == 0])
    # Based on your framework, filter images with more than 3 faces as multi-face images
    images_with_multiple_faces = len(df[df[face_count_col] > 3])

    success_rate = round((success_count / total_images * 100), 2) if total_images > 0 else 0


    stats = {
        "总图片数": total_images,
        "成功检测图片数": success_count,
        "检测成功率 (%)": success_rate,
        "检测到人脸总数": total_faces_detected,
        "平均每张图片人脸数": average_faces_per_image,
        "无脸图片数 (检测为0)": images_with_zero_faces,
        "多人脸图片数 (检测 > 3)": images_with_multiple_faces,
        "检测失败/异常数量": detection_failures
    }

    stats_file = os.path.join(Paths.ANALYSIS_DIR, "stats.txt")
    # print(f"保存统计数据到: {stats_file}") # Print in main function
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("=== 人脸识别测试结果统计 ===\n\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        # print("统计数据保存成功。") # Print in main function
    except Exception as e:
         print(f"[错误] 保存统计数据失败: {e}")
         traceback.print_exc()


    return stats

def plot_distributions(df):
    """绘制人脸数量分布直方图"""
    print("正在绘制分布直方图...")
    try:
        font_path = get_chinese_font()
        # Set matplotlib to support Chinese characters
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["axes.unicode_minus"] = False # Resolve negative sign display issue
    except FileNotFoundError as e:
         print(f"[警告] 未找到中文字体，分布图可能无法正确显示中文: {e}")
         # Fallback to a default font or skip plotting
         plt.rcParams["font.family"] = 'sans-serif'
         plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
         print(f"[警告] 设置matplotlib字体失败: {e}")
         traceback.print_exc()
         plt.rcParams["font.family"] = 'sans-serif'
         plt.rcParams["axes.unicode_minus"] = False


    # !!! Reference column name from loaded data header !!!
    face_count_col = "人脸数" # Modified based on latest Excel screenshot header
    if face_count_col not in df.columns:
         print(f"[WARNING] Loaded file is missing expected column '{face_count_col}', skipping distribution plot.")
         return

    plt.figure(figsize=(10, 6))
    # Plot histogram, bins=max_faces+1, so each bar represents a specific face count
    max_faces = int(df[face_count_col].max())
    if max_faces < 10: # If max face count is small, limit number of bins
        bins = range(0, max_faces + 2) # +2 ensures max value and right boundary are included
        xticks = range(0, max_faces + 1)
    else: # If max face count is large, use more bins or group appropriately
        bins = range(0, max_faces + 2)
        xticks = range(0, max_faces + 1, max(1, max_faces // 10)) # Display ticks every certain number

    # Adjust bin positions to align with integer values
    n, bins, patches = plt.hist(df[face_count_col], bins=bins, edgecolor="black", align='left', rwidth=0.8)

    plt.title("每张图片检测到人脸数量分布")
    plt.xlabel("检测到人脸数量")
    plt.ylabel("图片数量")
    plt.xticks(xticks) # Set x-axis ticks to integer face counts
    plt.grid(axis='y', alpha=0.75) # Add grid lines

    # Display counts above each bar (only for bars with count > 0)
    for i in range(len(n)):
         if n[i] > 0:
             # Calculate center position of the bar
             x_pos = bins[i] + (bins[i+1] - bins[i]) / 2
             plt.text(x_pos, n[i], str(int(n[i])), ha='center', va='bottom')


    plt.tight_layout() # Adjust layout to prevent label overlap
    vis_file = os.path.join(Paths.VIS_DIR, "face_distribution.png")
    # print(f"Saving distribution plot to: {vis_file}") # Print in main function
    try:
        plt.savefig(vis_file, dpi=300)
        # print("Distribution plot saved successfully.") # Print in main function
    except Exception as e:
        print(f"[ERROR] Failed to save distribution plot: {e}")
        traceback.print_exc()
    finally:
        plt.close()

def sample_images(df, n=10):
    """Randomly samples N images and copies their detected result images to sample directory"""
    print(f"正在随机抽取 {n} 张样本图片...")
    # !!! Reference column name from loaded data header !!!
    filename_col = "图片名" # Modified based on latest Excel screenshot header
    # This Excel format does not have an original image path column, so we won't try to get original path from the DataFrame

    if filename_col not in df.columns:
         print(f"[WARNING] Loaded file is missing expected column '{filename_col}', cannot sample images.")
         return

    # Ensure there are enough images to sample
    if len(df) < n:
        n = len(df)
        print(f"[TIP] Total number of images ({len(df)}) is less than requested ({n}), sampling all {n} images.")

    # Randomly sample row indices
    sample_indices = random.sample(range(len(df)), n)
    samples_df = df.iloc[sample_indices]

    copied_count = 0
    for index, row in samples_df.iterrows():
        original_filename = row[filename_col]
        # Detected result images are usually named "detected_" + original filename and placed in DETECTION_RESULTS_DIR
        src_image_name = f"detected_{original_filename}"
        src_path = os.path.join(Paths.DETECTION_RESULTS_DIR, src_image_name)
        dst_path = os.path.join(Paths.SAMPLE_CHECK_DIR, original_filename) # Copy to sample directory using original filename

        if os.path.exists(src_path):
            try:
                # Use PIL to copy image, better compatibility with various formats
                img = Image.open(src_path)
                img.save(dst_path) # PIL handles format on save
                copied_count += 1
                # print(f"Successfully copied detected image: {src_image_name}")
            except Exception as e:
                print(f"[WARNING] Could not process or copy sample image {src_image_name} to {dst_path}: {e}")
                traceback.print_exc()
        else:
             # !!! In this Excel format, there is no original image path, so we cannot try to copy original image !!!
             print(f"[WARNING] Sample image source file not found: {src_path}. Cannot copy original image as report file does not contain original path.")


    print(f"Successfully copied {copied_count} sample images to {Paths.SAMPLE_CHECK_DIR}")


def find_suspects(df):
    """Filters out suspicious detection results (e.g., too many faces or abnormal status)"""
    print("正在筛选可疑识别结果...")
    # !!! Reference column names from loaded data header !!!
    face_count_col = "人脸数" # Modified based on latest Excel screenshot header
    status_col = "状态"     # Modified based on latest Excel screenshot header
    # *** Based on the latest Excel screenshot, the successful status string is "成功" ***
    success_status = "成功"

    # Ensure column names exist
    if face_count_col not in df.columns or status_col not in df.columns:
         print(f"[WARNING] Loaded file is missing expected column names ('{face_count_col}' or '{status_col}'), skipping suspicious results filtering.")
         return pd.DataFrame() # Return empty DataFrame

    # Use cleaned status column for criteria
    status_col_for_criteria = status_col + '_cleaned'
    if status_col_for_criteria not in df.columns:
        print(f"[WARNING] Cleaned status column '{status_col_for_criteria}' not found, using original status column for filtering, results may be inaccurate.")
        status_col_for_criteria = status_col


    # Add print information to help understand filtering results
    print(f"--- Suspicious Results Filtering Details ---")
    print(f"Filtering criteria: ('{face_count_col}' > 5) OR ('{status_col_for_criteria}' != '{success_status}')")
    if status_col_for_criteria in df.columns:
         print(f"Unique values in column used for filtering ('{status_col_for_criteria}'): {df[status_col_for_criteria].unique()}")
    if face_count_col in df.columns:
         print(f"Partial statistics for column used for filtering ('{face_count_col}'):\n{df[face_count_col].describe()}")
    print(f"-------------------------")


    # Suspicious criteria: Face count > 5 (adjustable) OR Status is not "成功"
    # Use .copy() to avoid SettingWithCopyWarning
    suspects = df[(df[face_count_col] > 5) | (df[status_col_for_criteria] != success_status)].copy()

    output_path = os.path.join(Paths.SUSPECT_DIR, "suspect_results.csv")
    if not suspects.empty:
         # Use utf_8_sig encoding for better Chinese display in Excel when opening CSV
         try:
              suspects.to_csv(output_path, index=False, encoding="utf_8_sig")
              print(f"Found {len(suspects)} suspicious results, saved to: {output_path}")
              # Print first 5 suspicious results examples
              print("First 5 suspicious results examples:")
              # Temporarily rename columns for easier printing
              suspects_print = suspects.rename(columns={'图片名': '文件名', '人脸数': '人脸数量', '状态': '状态'})
              print(suspects_print.head().to_string())
              print("-" * 20)

         except Exception as e:
              print(f"[ERROR] Failed to save suspicious results to CSV: {e}")
              traceback.print_exc()
    else:
         print("No suspicious detection results found.")
         # Optionally, create an empty CSV file or do not generate one
         # try:
         #     pd.DataFrame(columns=df.columns).to_csv(output_path, index=False, encoding="utf_8_sig")
         #     print(f"No suspicious results found, generated empty suspicious results file: {output_path}")
         # except Exception as e:
         #     print(f"[WARNING] Could not generate empty suspicious results file: {e}")


    return suspects

def generate_pdf_report(stats, suspect_count):
    """Generates analysis report in PDF format"""
    print("正在生成 PDF 报告...")
    pdf = None # Initialize to None
    try:
        font_path = get_chinese_font()
        pdf = PDFReport(font_path)
        pdf.add_page()

        # === I. Basic Statistics ===
        pdf.set_font("Chinese", "", 14)
        pdf.cell(0, 10, "一、基础统计", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        pdf.ln(2) # Add a little space

        # Only draw stats table if stats are not empty and there is data
        if stats and stats.get("总图片数", 0) > 0:
            pdf.set_font("Chinese", "", 12)
            # Draw table
            # Table Header
            pdf.set_fill_color(230, 230, 230) # Set gray background
            pdf.cell(95, 10, "统计项", border=1, align='C', fill=True)
            pdf.cell(95, 10, "结果", border=1, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            pdf.set_fill_color(255, 255, 255) # Restore white background

            # Table Content
            for k, v in stats.items():
                 pdf.cell(95, 10, k, border=1)
                 pdf.cell(95, 10, str(v), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
             pdf.set_font("Chinese", "", 12)
             pdf.cell(0, 10, "统计数据生成失败或数据为空，无法显示。", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


        pdf.ln(10) # New line

        # === II. Detection Count Distribution Plot ===
        pdf.set_font("Chinese", "", 14)
        pdf.cell(0, 10, "二、识别数量分布图", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        pdf.ln(2)

        vis_image_path = os.path.join(Paths.VIS_DIR, "face_distribution.png")
        if os.path.exists(vis_image_path):
             try:
                 # Add image, w=180 width, height will be adjusted proportionally
                 pdf.image(vis_image_path, x=None, y=None, w=180)
             except Exception as e:
                  print(f"[ERROR] Failed to add distribution plot to PDF: {e}")
                  traceback.print_exc()
                  pdf.set_font("Chinese", "", 12)
                  pdf.cell(0, 10, "添加分布图失败。", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
             pdf.set_font("Chinese", "", 12)
             pdf.cell(0, 10, "未找到分布图文件，无法添加到报告。", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


        pdf.ln(10) # New line

        # === III. Suspicious Detection Analysis ===
        pdf.set_font("Chinese", "", 14)
        pdf.cell(0, 10, "三、可疑识别分析", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        pdf.ln(2)

        pdf.set_font("Chinese", "", 12)
        # Modify multi_cell here to avoid potential font width calculation issues
        pdf.multi_cell(0, 8, f"根据设定标准（例如：单张图片检测到人脸数超过 5 个，或检测状态显示异常），共检测出 {suspect_count} 张图片有可疑识别结果。", align="L")

        suspect_csv_path_str = os.path.join(Paths.SUSPECT_DIR, 'suspect_results.csv')
        # Attempt to display path information on two lines
        pdf.cell(0, 8, "详细的可疑图片列表已保存为 CSV 文件，请查阅:", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        # Use multi_cell to ensure long path wraps
        pdf.multi_cell(0, 8, suspect_csv_path_str, align="L")


        out_pdf = os.path.join(Paths.ANALYSIS_DIR, "analysis_report.pdf")
        pdf.output(out_pdf)
        print(f"PDF 报告生成成功: {out_pdf}")

    except FileNotFoundError as e:
        print(f"[ERROR] Failed to generate PDF, missing font file or other dependency: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"[ERROR] An unknown error occurred during PDF generation: {e}")
        traceback.print_exc()


# ================= Main Process =================
def main():
    print("=== Starting Face Detection Analysis Report Generation Process ===")
    print("📊 [Initialization] Building analysis environment...")
    try:
        setup_dirs()
        print("✅ Initialization complete.")
    except Exception as e:
        print(f"[FAILURE] Failed to initialize directories: {e}")
        traceback.print_exc()
        return

    df = pd.DataFrame() # Initialize as empty DataFrame to prevent errors if loading fails
    try:
        print("📥 [Loading] Reading detection report file...")
        df = load_data()
        print("✅ Data loaded successfully.")
    except (FileNotFoundError, IOError) as e:
        print(f"[FAILURE] Failed to load report file: {e}")
        print(f"Please check if Paths.REPORT_FILE ({Paths.REPORT_FILE}) points to the correct Excel file, and if you have read permissions for the file.")
        return # Stop execution if file loading fails
    except Exception as e:
        print(f"[FAILURE] An error occurred while loading the report file: {e}")
        traceback.print_exc()
        return # Stop execution for other loading errors


    stats = {} # Initialize as empty dictionary
    try:
        print("📈 [Analysis] Generating statistics...")
        stats = generate_stats(df)
        print("✅ Statistics generated successfully.")
    except ValueError as e:
         print(f"[FAILURE] Failed to generate statistics: {e}")
         print("Please check if the loaded file conforms to the expected format (e.g., if headers include '图片名', '人脸数', '状态'), and if the column names or success status string referenced in the code are correct.")
         traceback.print_exc()
    except Exception as e:
        print(f"[FAILURE] Failed to generate statistics: {e}")
        traceback.print_exc()


    try:
        print("📊 [Plotting] Drawing distribution histogram...")
        plot_distributions(df)
        print("✅ Distribution plot generated successfully.")
    except FileNotFoundError as e:
         print(f"[FAILURE] Failed to plot distribution, Chinese font not found: {e}")
         print("Please check the get_chinese_font function or manually specify a Chinese font path, or ensure Chinese fonts are installed on your system.")
    except Exception as e:
        print(f"[FAILURE] Failed to plot distribution: {e}")
        traceback.print_exc()


    try:
        print("📸 [Sampling] Sampling example images...")
        # Sample 20 images by default, you can change the value of n
        sample_images(df, n=20)
        print("✅ Sample images copied successfully.")
    except Exception as e:
        print(f"[FAILURE] Failed to sample images: {e}")
        traceback.print_exc()


    suspect_count = 0 # Initialize suspicious count
    try:
        print("🚨 [Filtering] Filtering suspicious detection results...")
        suspects = find_suspects(df)
        suspect_count = len(suspects)
        print("✅ Suspicious results filtering completed.")
    except ValueError as e:
         print(f"[FAILURE] Failed to filter suspicious results: {e}")
         print("Please check if the loaded file conforms to the expected format (e.g., if headers include '图片名', '人脸数', '状态'), and if the column names or success status string referenced in the code are correct.")
         traceback.print_exc()
    except Exception as e:
        print(f"[FAILURE] Failed to filter suspicious results: {e}")
        traceback.print_exc()
        # suspect_count remains 0


    # Attempt to generate PDF report regardless of whether previous steps were fully successful
    # PDF generation might depend on font and image files, exception handling is done inside generate_pdf_report
    try:
        print("📄 [Reporting] Generating PDF report...")
        # Only pass stats to PDF function if stats were successfully generated and not empty
        if stats and stats.get("总图片数", 0) > 0:
            generate_pdf_report(stats, suspect_count)
        else:
            print("[WARNING] Statistics generation failed or data is empty, PDF report will be missing the statistics table.")
            # Attempt to generate a PDF without the stats table
            generate_pdf_report({}, suspect_count) # Pass empty stats data
        print("✅ PDF report generation process attempted. Please check the output files.")
    except Exception as e:
        print(f"[FAILURE] Failed to generate PDF report: {e}")
        traceback.print_exc()


    print("\n=== Face Detection Analysis Report Generation Process Finished ===")
    print(f"All analysis results output directory: {Paths.ANALYSIS_DIR}")
    print("Please check the following files for detailed analysis results:")
    print(f" - 📑 Statistics text: {os.path.join(Paths.ANALYSIS_DIR, 'stats.txt')}")
    print(f" - 📊 Distribution plot: {os.path.join(Paths.VIS_DIR, 'face_distribution.png')}")
    print(f" - 🖼️ Sample images: {Paths.SAMPLE_CHECK_DIR}")
    # Only print path to suspicious results CSV if find_suspects ran successfully and found suspicious results
    if suspect_count > 0 and os.path.exists(os.path.join(Paths.SUSPECT_DIR, 'suspect_results.csv')):
         print(f" - 🚩 Suspicious results: {os.path.join(Paths.SUSPECT_DIR, 'suspect_results.csv')}")
    else:
         print(f" - 🚩 Suspicious results: No suspicious results found or filtering failed, {os.path.join(Paths.SUSPECT_DIR, 'suspect_results.csv')} was not generated.")

    pdf_report_path = os.path.join(Paths.ANALYSIS_DIR, 'analysis_report.pdf')
    if os.path.exists(pdf_report_path):
         print(f" - 📝 PDF report: {pdf_report_path}")
    else:
         print(f" - 📝 PDF report: PDF report generation failed, file {pdf_report_path} does not exist.")


if __name__ == "__main__":
    main()