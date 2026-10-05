"""Offline UI checks; no actual Groq API calls or model downloads."""
import ast
import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parent
source = (root / 'app.py').read_text(encoding='utf-8')
compile(source, 'app.py', 'exec')
documents = list((root / 'data').glob('*.txt'))
assert len(documents) >= 10
assert all(p.read_text(encoding='utf-8-sig').strip() for p in documents)
with (root / 'test_questions.csv').open(encoding='utf-8', newline='') as file:
    questions = list(csv.DictReader(file))
assert len(questions) >= 10
assert sum(q['answerable'] == 'false' for q in questions) >= 2
assert '.streamlit/secrets.toml' in (root / '.gitignore').read_text()

# Test the actual chunk and follow-up functions without launching the app.
tree = ast.parse(source)
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'chunk_text', 'resolve_question'}]
namespace = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app-functions', 'exec'), namespace)
parts = namespace['chunk_text']('0123456789' * 120)
assert parts[0][-150:] == parts[1][:150]
assert namespace['chunk_text']('') == []
history = [{'role': 'assistant', 'sources': [{'title': 'ผัดกะเพราไก่'}]}]
assert 'ผัดกะเพราไก่' in namespace['resolve_question']('แล้วใช้เวลากี่นาที?', history)
assert namespace['resolve_question']('ผัดไทยทำอย่างไร?', history) == 'ผัดไทยทำอย่างไร?'

class FakeModel:
    def encode(self, texts, **kwargs):
        return np.asarray([[0., 1.] if 'พิซซ่า' in text else [1., 0.] for text in texts], dtype='float32')

calls = []
class FakeGroq:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kwargs):
        calls.append(kwargs)
        content = 'ประมาณ 20 นาที [01_kaprao.txt]' if 'กี่นาที' in kwargs['messages'][-1]['content'] else 'เนื้อไก่ 300 กรัม [01_kaprao.txt]'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

with patch('sentence_transformers.SentenceTransformer', return_value=FakeModel()), patch('groq.Groq', FakeGroq):
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=30)
    app.secrets['GROQ_API_KEY'] = ''
    app.run()
    assert not app.exception
    assert app.chat_input[0].disabled
    app.secrets['GROQ_API_KEY'] = 'gsk_test_not_a_real_key'
    app.run()
    assert not app.exception and not app.chat_input[0].disabled
    app.chat_input[0].set_value('ผัดกะเพราไก่ใช้ไก่กี่กรัม?').run()
    assert not app.exception
    assert '300' in app.session_state['messages'][-1]['content']
    assert app.session_state['messages'][-1]['sources']
    app.chat_input[0].set_value('แล้วใช้เวลากี่นาที?').run()
    assert not app.exception
    assert '20' in app.session_state['messages'][-1]['content']
    assert 'ผัดกะเพราไก่' in calls[-1]['messages'][-1]['content']
    count = len(calls)
    app.chat_input[0].set_value('พิซซ่าใช้แป้งเท่าไร?').run()
    assert not app.exception
    assert app.session_state['messages'][-1]['content'] == 'ไม่พบข้อมูลในเอกสารความรู้'
    assert not app.session_state['messages'][-1]['sources']
    assert len(calls) == count
    app.button[0].click().run()
    assert not app.exception and not app.session_state['messages']
print('PASS: files, questions, chunk overlap, follow-up, missing key, chat, citations, refusal without API, clear chat.')