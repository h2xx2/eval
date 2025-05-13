import os
import json
import importlib
from evals.registry import Registry
from evals.record import RecorderBase

class CustomFileRecorder(RecorderBase):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "results.jsonl")
        self.matches = []

    def record_match(self, **kwargs):
        self.matches.append(kwargs)
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs) + "\n")

    def get_accuracy(self):
        if not self.matches:
            return 0.0
        correct = sum(1 for match in self.matches if match.get("correct"))
        return correct / len(self.matches)

    def dump(self):
        pass

# Настройка реестра
registry = Registry()
registry._evals = {
    "custom.chatbot_general_knowledge": {
        "cls": "chat_eval:ChatBotEval",
        "registry_path": "chat_eval/registry/evals/chatbot.yaml",
        "args": {
            "test_jsonl": "chat_eval/data/general_knowledge.jsonl"
        }
    }
}
registry._completion_fns = {
    "chatbot": {
        "cls": "chat_eval:ChatBotCompletionFn",
        "registry_path": "chat_eval/registry/completion_fns/chatbot.yaml"
    }
}

# Создаем рекордер для сохранения результатов
output_dir = "eval_results"
os.makedirs(output_dir, exist_ok=True)
recorder = CustomFileRecorder(output_dir)

# Загружаем оценку
eval_name = "custom.chatbot_general_knowledge"
eval_spec = registry.get_eval(eval_name)

# Динамически импортируем класс оценки
module_name, class_name = eval_spec.cls.rsplit(":", 1)
module = importlib.import_module(module_name)
eval_class = getattr(module, class_name)

# Создаем экземпляр оценки
eval_instance = eval_class(registry=registry, **eval_spec.args)

# Запускаем оценку
metrics = eval_instance.run(recorder)
print("Results:", metrics)

# Сохраняем результаты
recorder.dump()