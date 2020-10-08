# Robert Chatterton
# CS 4120
# 9/24/20

import re

class Chatbot:

    def __init__(self):
        # TODO: implement if needed
        # If you want to load in any lexicons, do it here and use relative file
        # paths. You'll upload these files as well.
        # Absolute file paths will break the autograder.
        # Let us know if you are using very large (> 1MB) files.         
        pass

    def getIndexDefault(self, _list, elem, default):
        try:
            itemAtIdx = _list[elem]
            return itemAtIdx
        except IndexError:
            return default
    
    def respond(self, userinput):
        """
        Prompts the chatbot to respond to the given string.
        Parameters:
            input - string utterance from the user to respond to
        Return: string bot response
        """
        
        # response word lists
        pronouns = ["you", "me", "i", "we", "us"]
        possPronouns = ["your", "my", "our"]
        toBe = ["are", "am", "is", "was", "were"]
        contractions = ["i'm", "you're", "we're"]
        conjunctions = ["and", "or", "but"]
        
        possPronounResponse = {
            "your":"my", "my":"your", "our":"your"
        }
        
        # if there is no user input
        if userinput == "":
            return "What did you say?"
        
        words = re.findall(r"[\w\"]+|[.,!?;]", userinput)
        
        # for basic math operations and checking where punctuation is
        firstPunc = -1
        firstFound = False
        for c in userinput:
            if c in "+-*/":
                return self.special_respond(userinput)
            elif c in ".!?" and not firstFound:
                firstPunc = words.index(c)
                firstFound = True
                
        length = len(words)
        
        newWords = []
        lastPronoun = ""
        lastConjunc = -1
         
        # guess at tense/person changes
        for word in range(len(words)):
            w = words[word].lower()
            
            # pronoun switching
            if w in pronouns:
                if w == "you":
                    if lastConjunc != -1:
                            if (float(word) - lastConjunc) / (length - lastConjunc) >= 0.5:
                                newWords.append("me")
                            else:
                                newWords.append("i")
                    elif firstPunc != -1: # assuming a period marks a sentence end
                        if word < firstPunc: # asking is this the first sentence?
                            if float(word) / firstPunc >= 0.5: # if "you" is after the halfway point in the sentence, output "i"
                                newWords.append("me")
                            else: # if before the halfway point, output "i"
                                newWords.append("i")
                        else:
                            if (float(word) - firstPunc) / (length - firstPunc) >= 0.5:
                                newWords.append("me")
                            else:
                                newWords.append("i")
                    else: # no periods in this user input, use length of input instead
                        if float(word) / length >= 0.5: 
                            newWords.append("me")
                        else:
                            newWords.append("i")
                else:
                    newWords.append("you")
                lastPronoun = w
            
            # possPronouns
            elif w in possPronouns:
                newWords.append(possPronounResponse.get(w, ""))
            
            # toBe
            elif w in toBe:
                if w == "are":
                    if lastPronoun == "you":
                        newWords.append("am")
                    elif self.getIndexDefault(words, (word + 1), "") == "you":
                        newWords.append("am")
                    else:
                        newWords.append("are")
                elif w == "am":
                    newWords.append("are") 
                elif w == "is":
                    newWords.append("is")
                elif w == "was":
                    if self.getIndexDefault(words, (word + 1), "") == "you":
                        newWords.append("were")
                    elif lastPronoun == "i" or lastPronoun == "me":
                        newWords.append("were")
                    else:
                        newWords.append("was")
                elif w == "were":
                    newWords.append("was")
            
            # contractions
            elif w in contractions: 
                if w == "i\'m":
                    newWords.append("you\'re")
                else:
                    newWords.append("i\'m")
            
            # marking conjunctions
            elif w in conjunctions:
                newWords.append(w)
                lastConjunc = word
                print(lastConjunc)
            
            # just a normal word
            else:
                newWords.append(w)
            prevWord = w                        
        
        # add back spaces
        output = ""
        i = 0
        for word in newWords:
            if word in ".,!?;\"" or i == 0:
                output = output + word
            else:
                output = output + " " + word
            i = i + 1
        return output
        
    def special_respond(self, userinput):
        """
        Prompts the chatbot to respond to the given string.
        Parameters:
            input - string utterance from the user to respond to
        Return: string bot response
        
        This special response is called when asked to perform an elementary math function 
            i.e. addition, subtraction, multiplication, and division
        """
        return userinput + " is " + str(eval(userinput))

    def greeting(self):
        """
        Prompts the chatbot to give an initial greeting.
        Return: string bot initial greeting
        """
        return "Hello, I am CHATBOT. Tell me something."

    def __str__(self):
        return "CHATBOT"

def main():
    # Create a new chatbot
    cb = Chatbot()
    # the chatbot always begins by greeting the user
    begin = cb.greeting()
    print(cb, ":", begin)
    user_input = input("> ")

    # Any case of writing the word "exit" will cause the program to stop
    while user_input.lower() != "exit":
        bot_phrase =  cb.respond(user_input)
        print(cb, ":", bot_phrase)
        user_input = input("> ")

    print("Goodbye!")


# This makes it so that the main function only runs when this file
# is directly run and not when it is imported as a module
if __name__ == "__main__":
    main()
