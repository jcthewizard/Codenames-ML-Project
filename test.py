import finetuner
def main():
    model = finetuner.build_model('jinaai/jina-embedding-t-en-v1')
    user_input_reds = input("Please enter all the red words in the form \"word1, word2, word3...\": ")
    user_input_blues = input("Please enter all the blue words in the form \"word1, word2, word3...\": ")
    user_input_neutrals = input("Please enter all the neutral words in the form \"word1, word2, word3...\": ")
    user_input_black = input("Please enter the death word: ")
    user_input_hint = input("Please enter your hint: ")
    blue = inputToArray(user_input_blues)
    red = inputToArray(user_input_reds)
    black = user_input_black
    neutral = inputToArray(user_input_neutrals)
    embeddings = finetuner.encode(
        model=model,
        data=getBoard(blue, red, black, neutral)
    )
    board = getBoard(blue, red, black, neutral)
    hint_embedding = finetuner.encode(model=model, data=[user_input_hint])[0]
    for i in range(len(embeddings)):
        print(user_input_hint + " and " + board[i] + ": " + str(finetuner.cos_sim(hint_embedding, embeddings[i])))
    blue_score = 0
    red_score = 0
    for j in range(len(blue)):
        res = finetuner.cos_sim(hint_embedding, embeddings[j])
        if res > 0:
            blue_score += res
    for k in range(len(blue), len(red)+len(blue)):
        res = finetuner.cos_sim(hint_embedding, embeddings[k])
        if res > 0:
            red_score += res
    black_score = finetuner.cos_sim(hint_embedding, embeddings[len(blue) + len(red)])
    print("red score: ", red_score)
    print("blue score: ", blue_score)
    if red_score > blue_score + .2 and black_score < .2:
        print("GOOD HINT FOR RED")
    elif red_score + .2 < blue_score and black_score < .2:
        print("GOOD HINT FOR BLUE")
    else:
        print("BAD")

def inputToArray(input):
    word_list = [word.strip() for word in input.split(',')]
    return word_list

def getBoard(blues, reds, black, neutrals):
    board = blues+reds+[black]+neutrals
    return board

if __name__ == "__main__":
    main()