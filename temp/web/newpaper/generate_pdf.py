import pdfkit , os
from jinja2 import Environment, FileSystemLoader

def render_html(template_name, context):
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))

    template = env.get_template(template_name)
    return template.render(context)

with open(r"C:\Users\boonh\Downloads\geo\_posts\web\newpaper\content.txt", "r",encoding='utf-8') as f:
    content = f.read()

context = {
    'title' : '中国金融市场' ,
    'description' : "中国金融市场概览及中美差异化对比" ,
    'content' : content
    # 'image' : r'C:\Users\boonh\Downloads\geo\_posts\web\newpaper\static\images\chart1.png'
}

html = render_html('index.html', context)

# print(html)
pdfkit.from_string(html, 'final_report.pdf')