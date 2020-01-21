import os


class Embedder():
    def __init__(self,parser,word_embedding_len,sentance_len):
        self.parser = parser
        self.sentance_len = sentance_len  # will use to determine the matrix dim
        self.word_embedding_len = word_embedding_len
        self.lang_model = None


    # main function to be used by the model trainer
    def game_state_to_image(self,game_state,game_variable):
        parsed_string = self.parser.parse_state(game_state, game_variable)
        return self.str_to_image(parsed_string)


    #this method will split the string we recieve from the parser, and remove unnecessary
    #tokens, such as :,;,, etc...
    def remove_and_parse_tokens(self,string_deque):
        deque_len = self.parser.state_len
        output_str = ""
        for i in range(0, deque_len):
            string = string_deque[i][:]
            if string[-1] == ".":
                string = string[:-1]
            output_str = string.replace(",", "")
            output_str = output_str.replace(":", "")
            output_str = output_str.replace(".", "")
            output_str = output_str.replace("!", "")
            output_str = output_str.split(" ")
            output_str = list(filter(None, output_str))
            output_str = [s.lower() for s in output_str]
            count = len(output_str) - self.sentance_len
            if count > 0: #turnicate sentance
                output_str = output_str[:self.sentance_len]
            # else: #padd sentance
            #     for _ in range(-count):
            #         output_str.append("pad")
        return output_str


    # recieves a "raw" game state string from the implemented parser and returns a
    # numpy 2d array of the sentance with dimenstions: [word_embedding_len,sentance_len]
    def str_to_image(self,state_string_deque):
        list_string = self.remove_and_parse_tokens(state_string_deque)
        text_image = self.lang_model[list_string].T
        return text_image

