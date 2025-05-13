import requests
from evals.api import CompletionFn, CompletionResult

class ChatBotCompletionFn(CompletionFn):
    def __init__(self, endpoint: str = "http://127.0.0.1:8000/get-course-info", **kwargs):
        self.endpoint = endpoint

    def __call__(self, prompt, **kwargs) -> CompletionResult:
        error_msg = None
        try:
            # Отправляем запрос к серверу
            response = requests.post(
                self.endpoint,
                json={"query": prompt},  # Отправляем запрос с данным prompt
                timeout=15
            )
            response.raise_for_status()  # Проверка на успешный ответ от сервера
            bot_response = response.json().get("response", "").strip()
        except requests.RequestException as e:
            error_msg = str(e)
            bot_response = f"Error: {error_msg}"

        # Создаем объект для результата
        result = ChatBotCompletionResult()  # Используем кастомный класс, который реализует `get_completions`
        result.completion = bot_response  # Ответ от бота
        result.metadata = {"error": error_msg}  # Метаданные с возможной ошибкой
        return result


class ChatBotCompletionResult(CompletionResult):
    def get_completions(self):
        # Возвращаем список возможных завершений
        return [self.completion]

