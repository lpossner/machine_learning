import logging

import tornado.ioloop
import tornado.web

from transformers import T5Tokenizer, T5ForConditionalGeneration


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

tokenizer = T5Tokenizer.from_pretrained(
    "google/flan-t5-small", clean_up_tokenization_spaces=True, legacy=False
)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


class ChatHandler(tornado.web.RequestHandler):

    def post(self):
        print("Here")
        user_input = self.get_json_argument("message")
        input_ids = tokenizer(user_input, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        response = tokenizer.decode(outputs[0])
        self.write({"response": response})

    def get_json_argument(self, name):
        try:
            return tornado.escape.json_decode(self.request.body)[name]
        except KeyError:
            raise tornado.web.HTTPError(400, f"Missing argument: {name}")


if __name__ == "__main__":
    api = tornado.web.Application(
        [
            (r"/chat", ChatHandler),
        ]
    )
    api.listen(80)
    logger.info("Server is running on http://localhost:80/chat")
    tornado.ioloop.IOLoop.current().start()
