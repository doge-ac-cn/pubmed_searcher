import sys
import requests
import time
import os
import datetime
import html
from bs4 import BeautifulSoup
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QSpinBox, QPushButton, QTextEdit, QProgressBar,
                               QListWidget, QMessageBox, QSizePolicy, QDialog,QScrollArea,QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer  # 添加了QTimer


# API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
LOG_DIR = "./pubmed_search_logs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

class SearchWorker(QThread):
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    update_keywords = pyqtSignal(list)
    search_complete = pyqtSignal(list)
    progress_stage = pyqtSignal(str)

    def __init__(self, params, api_token):
        super().__init__()
        self.params = params
        self.api_token = api_token
        self.running = True
        self.log_file = ""
        self.current_stage = ""

    def run(self):
        try:
            self.log_file = self.create_log_file()
            self.log_data(f"初始化完成 | 参数: {self.params}")
            self.main_process()
        except Exception as e:
            self.update_log.emit(f"错误: {str(e)}")

    def stop(self):
        self.running = False
        self.log_data("搜索已中止")

    def log_data(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.update_log.emit(log_entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def create_log_file(self):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.params['domain'][:20]}_{timestamp}.log".replace(" ", "_")
        return os.path.join(LOG_DIR, filename)

    def call_llm(self, prompt, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}",  # 使用传入的token
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 4096
            }
            response = requests.post(API_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            self.log_data(f"API调用失败: {str(e)}")
            return None

    def generate_keywords(self):
        domain = self.params['domain']
        prompt = f"""请为'{domain}'研究领域生成{self.params['num_groups']}组可以直接粘贴到PubMed进行搜索的英文关键词，
        每行一组关键词，不要编号和其他说明文字。"""
        self.log_data(f'已向deepseek提问 {prompt}')
        response = self.call_llm(prompt,model='deepseek-ai/DeepSeek-R1')
        if response:
            keywords = [kw.strip() for kw in response.split('\n') if kw.strip()]
            return keywords[:self.params['num_groups']]
        return []

    def search_pubmed(self, keyword):
        params = {
            "db": "pubmed",
            "term": f'({keyword}) AND ("{self.params["start_year"]}"[Date - Publication] : "{self.params["end_year"]}"[Date - Publication])',
            "retmode": "json",
            "retmax": self.params['per_group'],
            "sort": "relevance"
        }
        try:
            search_response = requests.get(PUBMED_SEARCH_URL, params=params)
            id_list = search_response.json()['esearchresult']['idlist']
            return id_list[:self.params['per_group']]
        except Exception as e:
            self.log_data(f"搜索失败: {str(e)}")
            return []

    def fetch_full_article(self, pmid):
        try:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            title_tag = soup.find("h1", class_="heading-title")
            title = title_tag.get_text(strip=True) if title_tag else "无标题"
            
            abstract_div = soup.find("div", class_="abstract-content selected")
            abstract = "\n".join([p.get_text(strip=True) for p in abstract_div.find_all("p")]) if abstract_div else "无摘要"
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "link": url
            }
        except Exception as e:
            self.log_data(f"获取文章失败 PMID:{pmid} - {str(e)}")
            return None

    def score_article(self, domain, article):
        prompt = f"""请根据以下研究领域和文章摘要进行评分（1-10分）：
        研究领域：{domain}
        标题：{article['title']}
        摘要：{article['abstract']}
        只返回数字不要其他内容："""
        response = self.call_llm(prompt)
        try:
            return float(response.strip())
        except:
            return 0

    def generate_summary(self, article):
        prompt = f"""用一句话概括以下研究内容：
        标题：{article['title']}
        摘要：{article['abstract']}
        要求：中文、50字以内、以"本研究"开头"""
        response = self.call_llm(prompt, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        return response.strip() if response else "无法生成内容概要"

    def main_process(self):
        # 生成关键词
        self.progress_stage.emit("生成关键词...")
        keywords = self.generate_keywords()
        self.update_keywords.emit(keywords)
        self.log_data(f"生成关键词完成 | 数量: {len(keywords)}")

        # 收集文章
        all_articles = {}
        total_count = 0
        self.progress_stage.emit("收集文献...")
        
        for idx, kw in enumerate(keywords):
            if not self.running:
                return
            self.log_data(f"正在处理关键词 [{idx+1}/{len(keywords)}]: {kw}")
            pmids = self.search_pubmed(kw)
            articles = []
            for pmid in pmids:
                if article := self.fetch_full_article(pmid):
                    articles.append(article)
                    time.sleep(0.5)
            
            for article in articles:
                total_count += 1
                if article['pmid'] not in all_articles:
                    all_articles[article['pmid']] = article
            self.update_progress.emit(int((idx+1)/len(keywords)*50))
            time.sleep(1)
        self.log_data(f"共获取到 {total_count} 篇文献")

        # 评分文章
        self.progress_stage.emit("评分文献...")
        scored_articles = []
        for idx, (pmid, article) in enumerate(all_articles.items()):
            if not self.running:
                return
            article['score'] = self.score_article(self.params['domain'], article)
            self.log_data(f"评分完成 | PMID: {pmid} - 评分: {article['score']} - 标题: {article['title']}")
            scored_articles.append(article)
            self.update_progress.emit(50 + int((idx+1)/len(all_articles)*40))
            time.sleep(0.5)

        # 生成摘要
        self.progress_stage.emit("生成摘要...")
        final_articles = []
        for idx, article in enumerate(scored_articles):
            if not self.running:
                return
            article['summary'] = self.generate_summary(article)
            self.log_data(f"生成摘要完成 | PMID: {article['pmid']} - 评分: {article['score']}  - 摘要: {article['summary']}")
            final_articles.append(article)
            self.update_progress.emit(90 + int((idx+1)/len(scored_articles)*10))
            time.sleep(0.5)

        # 最终结果
        sorted_articles = sorted(final_articles, key=lambda x: x['score'], reverse=True)
        filtered_articles = [x for x in sorted_articles if x['score'] >= 6.0][:30]
        self.search_complete.emit(filtered_articles)
        self.update_progress.emit(100)

class WelcomeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("欢迎使用")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # 移除帮助按钮
        layout = QVBoxLayout()
        
        # 创建提示信息标签
        message = QLabel()
        message.setText("""<b>请按以下步骤获取API密钥：</b>
        <ol>
            <li>前往<a href="https://cloud.siliconflow.cn/i/XOLuRqkK">硅基流动官网</a>注册账号</li>
            <li>登录后进入控制台，点击【API密钥】菜单</li>
            <li>创建新的API密钥并复制</li>
            <li>返回本程序粘贴到API Token输入框即可使用</li><br><br>
            <strong>分享该程序时，建议点击清空配置，防止自己的api泄露</strong>
        </ol>""")
        message.setOpenExternalLinks(True)  # 允许打开外部链接
        message.setTextFormat(Qt.RichText)
        message.setWordWrap(True)
        
        # 创建按钮
        close_btn = QPushButton("知道了")
        close_btn.clicked.connect(self.accept)
        
        # 布局设置
        layout.addWidget(message)
        layout.addWidget(close_btn)
        self.setLayout(layout)
        self.resize(400, 200)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PubMed文献搜索工具")
        self.setGeometry(100, 100, 1000, 800)
        self.search_thread = None
        self.init_ui()
        self.load_settings()
        
        # 在窗口初始化完成后显示欢迎弹窗
        QTimer.singleShot(100, self.show_welcome_dialog)  # 延迟100ms确保主窗口先显示

    def show_welcome_dialog(self):
        dialog = WelcomeDialog(self)
        dialog.exec_()

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        # 修改API Token输入区域
        api_layout = QHBoxLayout()
        self.add_input_field(api_layout, "硅基流动 API Token:", QLineEdit(), 'api_token_input')
        self.api_token_input.setEchoMode(QLineEdit.Password)
        
        # 添加清空配置按钮
        clear_btn = QPushButton("清空配置", clicked=self.clear_settings)
        api_layout.addWidget(clear_btn)
        
        layout.addLayout(api_layout)

        # 研究主题单独一行
        domain_layout = QHBoxLayout()
        self.add_input_field(domain_layout, "研究主题:", QLineEdit(), 'domain_input')
        self.domain_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addLayout(domain_layout)

        # 其他参数
        input_layout = QHBoxLayout()
        self.add_spinbox(input_layout, "起始年份:", 1900, datetime.datetime.now().year, datetime.datetime.today().year-6, 'start_year')
        self.add_spinbox(input_layout, "结束年份:", 1900, datetime.datetime.now().year, datetime.datetime.today().year, 'end_year')
        self.add_spinbox(input_layout, "每组关键词搜索的文章数量:", 1, 100, 10, 'per_group')
        self.add_spinbox(input_layout, "关键词组数:", 1, 50, 5, 'num_groups')
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始搜索", clicked=self.start_search)
        self.stop_btn = QPushButton("停止", clicked=self.stop_search)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        input_layout.addLayout(btn_layout)
        layout.addLayout(input_layout)

        # 关键词列表
        self.keyword_list = QListWidget()
        layout.addWidget(QLabel("推荐关键词:"))
        layout.addWidget(self.keyword_list)

        # 日志区域
        self.log_output = QTextEdit(readOnly=True)
        layout.addWidget(QLabel("实时日志:"))
        layout.addWidget(self.log_output)

        # 进度显示
        self.progress_bar = QProgressBar()
        self.stage_label = QLabel("准备就绪")
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress_bar)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def load_settings(self):
        settings = QSettings("MyApp", "PubMedSearcher")
        self.api_token_input.setText(settings.value("api_token", ""))

    def save_settings(self):
        settings = QSettings("MyApp", "PubMedSearcher")
        settings.setValue("api_token", self.api_token_input.text())


    def add_input_field(self, layout, label, widget, name):
        layout.addWidget(QLabel(label))
        setattr(self, name, widget)
        layout.addWidget(widget)

    def add_spinbox(self, layout, label, min_val, max_val, default, name):
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        self.add_input_field(layout, label, spinbox, name)
    def clear_settings(self):
        # 确认对话框
        reply = QMessageBox.question(
            self,
            '确认清空',
            '确定要清空所有保存的配置吗？',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 清除保存的配置
            settings = QSettings("MyApp", "PubMedSearcher")
            settings.clear()
            
            # 重置界面元素
            self.api_token_input.clear()
            self.domain_input.clear()
            self.start_year.setValue(2019)
            self.end_year.setValue(2025)
            self.per_group.setValue(10)
            self.num_groups.setValue(5)
            
            QMessageBox.information(self, "提示", "配置已重置为默认值")

    def start_search(self):
        if self.search_thread and self.search_thread.isRunning():
            QMessageBox.warning(self, "警告", "当前有搜索正在进行中")
            return

        if not self.api_token_input.text():
            QMessageBox.critical(self, "错误", "必须填写API Token")
            return
        self.save_settings()  # 保存API Token

        params = {
            'domain': self.domain_input.text(),
            'start_year': self.start_year.value(),
            'end_year': self.end_year.value(),
            'per_group': self.per_group.value(),
            'num_groups': self.num_groups.value()
        }

        if not params['domain']:
            QMessageBox.critical(self, "错误", "必须填写研究主题")
            return

        self.search_thread = SearchWorker(params, self.api_token_input.text())
        self.search_thread.update_log.connect(self.log_output.append)
        self.search_thread.update_progress.connect(self.progress_bar.setValue)
        self.search_thread.update_keywords.connect(self.update_keywords)
        self.search_thread.search_complete.connect(self.show_results)
        self.search_thread.progress_stage.connect(self.stage_label.setText)
        self.search_thread.start()


    def show_results(self, results):
        dialog = QDialog(self)
        dialog.setWindowTitle("搜索结果")
        dialog.setMinimumSize(600, 400)

        # 主布局
        layout = QVBoxLayout(dialog)

        # 滚动区域
        scroll = QScrollArea()
        content = QWidget()
        content_layout = QVBoxLayout(content)

        # 显示结果
        for idx, item in enumerate(results[:10], 1):
            # 转义HTML字符
            title = html.escape(item['title'])
            link = html.escape(item['link'])
            summary = html.escape(item['summary'])

            # 单个结果条目
            entry = QLabel()
            entry.setTextFormat(Qt.RichText)
            entry.setOpenExternalLinks(True)
            entry.setText(
                f"{idx}. [{item['score']:.1f}分]<br/>"
                f"<a href='{link}'>{title}</a><br/>"
                f"摘要：{summary}<br/>"
                f"链接：<a href='{link}'>{link}</a>"
            )
            content_layout.addWidget(entry)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # 底部按钮
        btn_box = QDialogButtonBox()
        copy_btn = QPushButton("复制")
        copy_btn.clicked.connect(lambda: self.copy_results(results))
        btn_box.addButton(copy_btn, QDialogButtonBox.ActionRole)
        btn_box.addButton(QDialogButtonBox.Close)
        btn_box.rejected.connect(dialog.reject)

        layout.addWidget(btn_box)
        dialog.exec_()

    def copy_results(self, results):
        text = ""
        for idx, item in enumerate(results[:10], 1):
            text += f"{idx}. [{item['score']:.1f}分] {item['title']}\n"
            text += f"摘要：{item['summary']}\n"
            text += f"链接：{item['link']}\n\n"
        QApplication.clipboard().setText(text.strip())

    def stop_search(self):
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.stop()
            self.log_output.append("用户请求停止搜索...")

    def update_keywords(self, keywords):
        self.keyword_list.clear()
        self.keyword_list.addItems(keywords)


    def closeEvent(self, event):
        if self.search_thread and self.search_thread.isRunning():
            reply = QMessageBox.question(
                self, '后台任务运行中',
                "确定要退出吗？当前搜索尚未完成",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.search_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
