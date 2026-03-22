#!/usr/bin/env python3
"""
生成搜索SKU特征排序实验结果报告PPTX
"""

import tempfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# 配色方案
COLORS = {
    'primary': RGBColor(46, 80, 144),      # 深蓝
    'secondary': RGBColor(91, 155, 213),   # 浅蓝
    'accent': RGBColor(237, 125, 49),      # 橙色
    'text': RGBColor(51, 51, 51),          # 深灰
    'light_gray': RGBColor(242, 242, 242), # 浅灰
    'white': RGBColor(255, 255, 255),
}

# 图表颜色
CHART_COLORS = ['#2E5090', '#5B9BD5', '#ED7D31', '#70AD47']

# 实验数据
data_sku_click_rate = pd.DataFrame({
    '分组': ['V1', 'V2\n(基准)', 'V3', 'V4'],
    'SKU点击率': [5.13, 5.33, 5.72, 5.63],
    'vs基准': [-3.85, 0, 7.23, 5.55]
})

data_click_position = pd.DataFrame({
    '分组': ['V1', 'V2\n(基准)', 'V3', 'V4'],
    '平均点击位置': [1.33, 1.42, 1.23, 1.29],
})

data_product_view = pd.DataFrame({
    '分组': ['V1', 'V2\n(基准)', 'V3', 'V4'],
    '商品查看次数': [22.21, 22.78, 21.67, 21.08],
})

data_purchase = pd.DataFrame({
    '分组': ['V1', 'V2\n(基准)', 'V3', 'V4'],
    '唤起购买次数': [1.50, 1.54, 1.51, 1.46],
})

data_comparison = pd.DataFrame({
    '指标': ['SKU点击率\n(正向)', '平均点击位置\n(正向)', '商品查看\n(负向)', '唤起购买\n(负向)'],
    'V3': [7.23, -13.57, -4.86, -1.81],
    'V4': [5.55, -9.10, -7.43, -5.12]
})


def generate_chart_image(chart_type, data, x_col, y_col=None, y_cols=None, title=None):
    """生成图表图片"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if chart_type == 'column':
        if y_cols:
            # 多系列柱状图
            x = data[x_col]
            width = 0.35
            for i, col in enumerate(y_cols):
                offset = (i - 0.5) * width
                bars = ax.bar([j + offset for j in range(len(x))], data[col], 
                             width, label=col, color=CHART_COLORS[i])
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=10)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x, fontsize=11)
            ax.legend(fontsize=11)
        else:
            # 单系列柱状图
            bars = ax.bar(data[x_col], data[y_col], color=CHART_COLORS[0])
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
    
    # 美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(temp_file.name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return temp_file.name


def add_title(slide, title):
    """添加幻灯片标题"""
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # 下划线装饰
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.15), Inches(2), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = COLORS['accent']
    line.line.fill.background()


def main():
    # 创建演示文稿
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # 使用空白布局
    blank_layout = prs.slide_layouts[6]
    
    # ==================== 1. 标题页 ====================
    slide = prs.slides.add_slide(blank_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.333), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "搜索使用SKU特征排序"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    p.alignment = PP_ALIGN.CENTER
    
    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.333), Inches(0.8))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = "A/B测试实验结果报告"
    p.font.size = Pt(24)
    p.font.color.rgb = COLORS['text']
    p.alignment = PP_ALIGN.CENTER
    
    date_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(12.333), Inches(0.5))
    tf = date_box.text_frame
    p = tf.paragraphs[0]
    p.text = "2026-03-05 ~ 2026-03-10"
    p.font.size = Pt(14)
    p.font.color.rgb = COLORS['secondary']
    p.alignment = PP_ALIGN.CENTER
    
    # ==================== 2. 实验概况 ====================
    slide = prs.slides.add_slide(blank_layout)
    add_title(slide, "实验概况")
    
    content_box = slide.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(11.833), Inches(5.5))
    tf = content_box.text_frame
    tf.word_wrap = True
    
    bullets = [
        "实验时间：2026-03-05 ~ 2026-03-10（6天）",
        "分组策略：V1/V2 对照组，V3/V4 实验组",
        "分流维度：搜索频次（top400 / top400以外）",
        "显著性标准：P-value ≤ 0.05",
        "日均实验UV：top400 ~1,250人 / top400以外 ~550人"
    ]
    
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = "• " + bullet
        p.font.size = Pt(22)
        p.font.color.rgb = COLORS['text']
        p.space_before = Pt(16)
    
    # ==================== 3. 核心指标分析 分隔页 ====================
    slide = prs.slides.add_slide(blank_layout)
    
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.8), Inches(13.333), Inches(2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLORS['primary']
    shape.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.4), Inches(12.333), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "核心指标分析"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER
    
    # ==================== 4. SKU点击率 ====================
    slide = prs.slides.add_slide(blank_layout)
    add_title(slide, "1. SKU点击率（越高越好）")
    
    chart_path = generate_chart_image('column', data_sku_click_rate, '分组', 'SKU点击率')
    slide.shapes.add_picture(chart_path, Inches(0.75), Inches(1.5), width=Inches(8))
    Path(chart_path).unlink()
    
    # 右侧洞察
    insight_box = slide.shapes.add_textbox(Inches(9), Inches(2), Inches(3.8), Inches(4))
    tf = insight_box.text_frame
    tf.word_wrap = True
    
    insights = [
        ("✅ V3", "点击率提升7.23%", "P=0.0299 显著"),
        ("⚠️ V4", "点击率提升5.55%", "P=0.0918 接近显著"),
    ]
    
    for i, (status, title_text, desc) in enumerate(insights):
        if i > 0:
            p = tf.add_paragraph()
            p.text = ""
            p.space_before = Pt(20)
        
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = status
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = COLORS['accent'] if '✅' in status else COLORS['secondary']
        
        p = tf.add_paragraph()
        p.text = title_text
        p.font.size = Pt(16)
        p.font.color.rgb = COLORS['text']
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(14)
        p.font.color.rgb = COLORS['secondary']
    
    # ==================== 5. 平均点击位置 ====================
    slide = prs.slides.add_slide(blank_layout)
    add_title(slide, "2. 平均点击位置（越小越好）")
    
    chart_path = generate_chart_image('column', data_click_position, '分组', '平均点击位置')
    slide.shapes.add_picture(chart_path, Inches(0.75), Inches(1.5), width=Inches(8))
    Path(chart_path).unlink()
    
    insight_box = slide.shapes.add_textbox(Inches(9), Inches(2), Inches(3.8), Inches(4))
    tf = insight_box.text_frame
    tf.word_wrap = True
    
    insights = [
        ("⚠️ V3", "位置提前13.57%", "P=0.0671 接近显著"),
        ("— V4", "位置提前9.10%", "P=0.253 不显著"),
    ]
    
    for i, (status, title_text, desc) in enumerate(insights):
        if i > 0:
            p = tf.add_paragraph()
            p.text = ""
            p.space_before = Pt(20)
        
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = status
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = COLORS['secondary']
        
        p = tf.add_paragraph()
        p.text = title_text
        p.font.size = Pt(16)
        p.font.color.rgb = COLORS['text']
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(14)
        p.font.color.rgb = COLORS['secondary']
    
    # ==================== 6. 商品查看次数 ====================
    slide = prs.slides.add_slide(blank_layout)
    add_title(slide, "3. 商品查看次数（浏览深度）")
    
    chart_path = generate_chart_image('column', data_product_view, '分组', '商品查看次数')
    slide.shapes.add_picture(chart_path, Inches(0.75), Inches(1.5), width=Inches(8))
    Path(chart_path).unlink()
    
    insight_box = slide.shapes.add_textbox(Inches(9), Inches(2), Inches(3.8), Inches(4))
    tf = insight_box.text_frame
    tf.word_wrap = True
    
    insights = [
        ("❌ V3", "下降4.86%", "P=0.0126 显著"),
        ("❌ V4", "下降7.43%", "P=0.0001 高度显著"),
    ]
    
    for i, (status, title_text, desc) in enumerate(insights):
        if i > 0:
            p = tf.add_paragraph()
            p.text = ""
            p.space_before = Pt(20)
        
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = status
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = RGBColor(192, 0, 0)  # 红色
        
        p = tf.add_paragraph()
        p.text = title_text
        p.font.size = Pt(16)
        p.font.color.rgb = COLORS['text']
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(14)
        p.font.color.rgb = COLORS['secondary']
    
    # ==================== 7. 唤起购买次数 ====================
    slide = prs.slides.add_slide(blank_layout)
    add_title(slide, "4. 唤起购买次数（直接关联GMV）")
    
    chart_path = generate_chart_image('column', data_purchase, '分组', '唤起购买次数')
    slide.shapes.add_picture(chart_path, Inches(0.75), Inches(1.5), width=Inches(8))
    Path(chart_path).unlink()
    
    insight_box = slide.shapes.add_textbox(Inches(9), Inches(2), Inches(3.8), Inches(4))
    tf = insight_box.text_frame
    tf.word_wrap = True
    
    insights = [
        ("— V3", "下降1.81%", "P=0.4301 不显著"),
        ("❌ V4", "下降5.12%", "P=0.0239 显著"),
    ]
    
    for i, (status, title_text, desc) in enumerate(insights):
        if i > 0:
            p = tf.add_paragraph()
            p.text = ""
            p.space_before = Pt(20)
        
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = status
        p.font.size = Pt(18)
        p.font.bold = True
        if '❌' in status:
            p.font.color.rgb = RGBColor(192, 0, 0)
        else:
            p.font.color.rgb = COLORS['secondary']
        
        p = tf.add_paragraph()
        p.text = title_text
        p.font.size = Pt(16)
        p.font.color.rgb = COLORS['text']
        
        p = tf.add_paragraph()
        p.text = desc
        p.font.size = Pt(14)
        p.font.color.rgb = COLORS['secondary']
    
    # ==================== 8. V3 vs V4 对比分隔页 ====================
    slide = prs.slides.add_slide(blank_layout)
    
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.8), Inches(13.333), Inches(2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLORS['primary']
    shape.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.4), Inches(12.333), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "V3 vs V4 策略对比"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER
    
    # ==================== 9. 对比图表 ====================
    slide = prs.slides.add_slide(blank_layout)
    add_title(slide, "V3 vs V4 变化幅度对比（%）")
    
    chart_path = generate_chart_image('column', data_comparison, '指标', y_cols=['V3', 'V4'])
    slide.shapes.add_picture(chart_path, Inches(0.5), Inches(1.5), width=Inches(9))
    Path(chart_path).unlink()
    
    insight_box = slide.shapes.add_textbox(Inches(9.5), Inches(1.8), Inches(3.3), Inches(5))
    tf = insight_box.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "关键发现"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    findings = [
        "SKU点击率：V3 > V4",
        "点击位置：V3 > V4",
        "商品查看：V3副作用更小",
        "唤起购买：V3稳定，V4伤GMV"
    ]
    
    for finding in findings:
        p = tf.add_paragraph()
        p.text = "• " + finding
        p.font.size = Pt(14)
        p.font.color.rgb = COLORS['text']
        p.space_before = Pt(8)
    
    # ==================== 10. 结论分隔页 ====================
    slide = prs.slides.add_slide(blank_layout)
    
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.8), Inches(13.333), Inches(2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLORS['primary']
    shape.line.fill.background()
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.4), Inches(12.333), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "结论与建议"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER
    
    # ==================== 11. 核心结论 ====================
    slide = prs.slides.add_slide(blank_layout)
    add_title(slide, "核心结论")
    
    # 左侧 - V3正向
    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(5.8), Inches(5))
    tf = left_box.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "✅ V3策略：正向效果显著"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLORS['accent']
    
    v3_items = [
        "SKU点击率 +7.23%（P=0.0299）",
        "点击位置提前13.57%（P=0.0671）",
        "唤起购买稳定（P=0.4301）"
    ]
    for item in v3_items:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(16)
        p.font.color.rgb = COLORS['text']
        p.space_before = Pt(12)
    
    # 右侧 - V4负向
    right_box = slide.shapes.add_textbox(Inches(7), Inches(1.6), Inches(5.8), Inches(5))
    tf = right_box.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "❌ V4策略：副作用明显"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = RGBColor(192, 0, 0)
    
    v4_items = [
        "商品查看 -7.43%（P<0.0001）",
        "加购 -6.51%（P=0.0053）",
        "唤起购买 -5.12%（P=0.0239）"
    ]
    for item in v4_items:
        p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(16)
        p.font.color.rgb = COLORS['text']
        p.space_before = Pt(12)
    
    # ==================== 12. 行动建议 ====================
    slide = prs.slides.add_slide(blank_layout)
    add_title(slide, "行动建议")
    
    content_box = slide.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(11.833), Inches(5.5))
    tf = content_box.text_frame
    tf.word_wrap = True
    
    recommendations = [
        ("【高优先级】上线V3策略", "正向效果显著，副作用可控"),
        ("【高优先级】暂缓V4策略", "多维度显著下降，风险较大"),
        ("【中优先级】延长实验周期", "当前仅6天，建议延长至2周确认趋势"),
        ("【后续优化】平衡精准与探索", "提升精准度的同时保留发现新商品的机会")
    ]
    
    for i, (title_text, desc) in enumerate(recommendations):
        if i > 0:
            p = tf.add_paragraph()
            p.text = ""
            p.space_before = Pt(8)
        
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = COLORS['primary']
        
        p = tf.add_paragraph()
        p.text = "   " + desc
        p.font.size = Pt(16)
        p.font.color.rgb = COLORS['text']
    
    # ==================== 13. Thank You ====================
    slide = prs.slides.add_slide(blank_layout)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(3), Inches(12.333), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Thank You"
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    p.alignment = PP_ALIGN.CENTER
    
    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(12.333), Inches(0.8))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Questions?"
    p.font.size = Pt(24)
    p.font.color.rgb = COLORS['secondary']
    p.alignment = PP_ALIGN.CENTER
    
    # 保存
    output_path = Path("/Users/pengtang/Documents/github/aliyun-devops/搜索SKU特征排序实验报告.pptx")
    prs.save(str(output_path))
    print(f"✅ PPTX已生成: {output_path}")


if __name__ == "__main__":
    main()