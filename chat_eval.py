from evals.eval import Eval
from evals.record import RecorderBase
from evals.registry import Registry
import importlib
from openai import OpenAI
import json

class ChatBotEval(Eval):
    def __init__(self, registry: Registry, **kwargs):
        # Подключение нужного модуля для обработки завершений
        module_name, class_name = "chat_eval.chat_completion:ChatBotCompletionFn".split(":")
        module = importlib.import_module(module_name)
        ChatBotCompletionFn = getattr(module, class_name)
        completion_fn = ChatBotCompletionFn()
        completion_fns = [completion_fn]

        # Путь к YAML файлу с настройками
        eval_registry_path = "chat_eval/registry/evals/chatbot.yaml"
        samples_jsonl = kwargs.get("test_jsonl")  # Путь к данным для тестирования

        super().__init__(completion_fns=completion_fns, eval_registry_path=eval_registry_path,
                         samples_jsonl=samples_jsonl, registry=registry)

        self._my_completion_fn = completion_fn
        self.test_jsonl = kwargs.get("test_jsonl")
        # Инициализация клиента OpenAI для оценки
        self.openai_client = OpenAI()  # Требует OPENAI_API_KEY в переменной окружения

    def eval_sample(self, sample, rng, recorder: RecorderBase):
        prompt = sample["input"]  # Запрос
        ideal_answer = sample["ideal"]  # Идеальный ответ
        sample_id = sample.get("sample_id", id(sample))  # Извлекаем sample_id

        result = self._my_completion_fn(prompt)  # Получаем ответ от бота
        bot_answer = result.get_completions()[0]  # Получаем ответ (первое завершение)

        # Оценка семантической схожести с помощью GPT, без нормализации
        eval_prompt = f"""
        Оцените, насколько ответ чат-бота семантически эквивалентен ожидаемому ответу.
        Ответ чат-бота: {bot_answer}
        Ожидаемый ответ: {ideal_answer}
        Критерии: релевантность (соответствие смыслу), связность (ясность и логичность).
        Учитывайте, что эмодзи (например, ❌) и небольшие различия в формулировках не влияют на семантическую эквивалентность.
        Верните JSON-объект:
        {{
            "is_correct": 1 если ответы эквивалентны, 0 если нет,
            "similarity": число от 0 до 1, отражающее степень схожести
        }}
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Используем gpt-4o
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"}
            )
            eval_result = response.choices[0].message.content
            eval_data = json.loads(eval_result)
            is_correct = eval_data.get("is_correct", 0) == 1
            similarity = eval_data.get("similarity", 0.0)
        except Exception as e:
            print(f"Ошибка при оценке GPT: {e}")
            # Запасной вариант: точное сравнение строк
            is_correct = bot_answer == ideal_answer
            similarity = 1.0 if is_correct else 0.0

        # Записываем результат в recorder, включая similarity в данные
        self.record_match(
            correct=is_correct,
            sampled=bot_answer,
            expected=ideal_answer,
            picked=bot_answer,
            prompt=prompt,
            sample_id=sample_id,
            similarity=similarity,
            recorder=recorder
        )

        # Дополнительное логирование
        print(f"Prompt: {prompt}")
        print(f"Expected: {ideal_answer}")
        print(f"Sampled: {bot_answer}")
        print(f"Correct: {is_correct}")
        print(f"Similarity: {similarity}")
        print(f"Picked: {bot_answer}")
        print(f"Sample ID: {sample_id}")
        print("-" * 40)

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()  # Получаем тестовые примеры
        for sample in samples:
            self.eval_sample(sample, None, recorder)

        # Для отладки: выводим события
        events = recorder.get_events("match")
        print("Events for 'match':", events)

        # Вычисляем точность и среднюю схожесть
        correct = sum(1 for e in events if e.data.get("correct", False))
        total = len(events)
        accuracy = correct / total if total > 0 else 0.0
        avg_similarity = sum(e.data.get("similarity", 0.0) for e in events) / total if total > 0 else 0.0

        # Для отладки: выводим промежуточные значения
        print("Correct:", correct)
        print("Total:", total)
        print("Computed accuracy:", accuracy)
        print("Average similarity:", avg_similarity)

        # Возвращаем метрики
        return {
            "accuracy": accuracy,
            "avg_similarity": avg_similarity
        }

    def record_match(self, correct, sampled, expected, picked, prompt, sample_id, similarity, recorder: RecorderBase):
        """Метод для записи результатов выполнения теста"""
        # Формируем данные события
        data = {
            "correct": correct,
            "expected": expected,
            "sampled": sampled,
            "prompt": prompt,
            "picked": picked,
            "sample_id": sample_id,
            "similarity": similarity
        }
        try:
            recorder.record_match(**data)
        except AttributeError:
            recorder.record(type="match", data=data)