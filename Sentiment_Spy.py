import colorama
from colorama import Fore, Style
from textblob import TextBlob

# 1) Initialize colorama for colored output
colorama.init()

# 2) Print an initial greeting with emojis and color
print(f"{Fore.CYAN}👋🎉 Welcome to Sentiment Spy! 🕵️{Style.RESET_ALL}")

# 3) Prompt the user for their name
user_name = input(f"{Fore.MAGENTA}Please enter your name: {Style.RESET_ALL} ").strip()
if not user_name:
    user_name = "Mystery Agent"  # If the user didn't type anything, use a default name

# 4) A list to store all conversation entries (text, polarity, sentiment_type)
conversation_history = []

# 5) Print instructions and an initial greeting to the user
print(f"\n{Fore.CYAN}Hello, Agent {user_name}!")
print(f"I will analyze your sentences with TextBlob and show you the sentiment. 🔍")
print(f"Type {Fore.YELLOW}'reset'{Fore.CYAN}, {Fore.YELLOW}'history'{Fore.CYAN}, "
      f"or {Fore.YELLOW}'exit'{Fore.CYAN} to quit.{Style.RESET_ALL}\n")

# 6) Main loop for user input
while True:
    user_input = input(f"{Fore.GREEN}>> {Style.RESET_ALL}").strip()

    # 6.1) If the user just pressed enter (empty input), prompt again
    if not user_input:
        print(f"{Fore.RED}Please enter some text or a valid command.{Style.RESET_ALL}")
        continue

    # 6.2) Exit command
    if user_input.lower() == "exit":
        print(f"\n{Fore.BLUE}🚪 Exiting Sentiment Spy. Farewell, Agent {user_name}! 🏁{Style.RESET_ALL}")
        break

    # 6.3) Reset command
    elif user_input.lower() == "reset":
        conversation_history.clear()
        print(f"{Fore.CYAN}🎉 All conversation history cleared!{Style.RESET_ALL}")
        continue

    # 6.4) History command
    elif user_input.lower() == "history":
        if not conversation_history:
            print(f"{Fore.YELLOW}No conversation history yet.{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}📜 Conversation History:{Style.RESET_ALL}")
            # Loop through all saved messages
            for idx, (text, polarity, sentiment_type) in enumerate(conversation_history, start=1):
                # Assign color & emoji based on sentiment type
                if sentiment_type == "Positive":
                    color = Fore.GREEN
                    emoji = "😊"
                elif sentiment_type == "Negative":
                    color = Fore.RED
                    emoji = "😢"
                else:
                    color = Fore.YELLOW
                    emoji = "😐"

                print(f"{idx}. {color}{emoji} {text} "
                      f"(Polarity: {polarity:.2f}, {sentiment_type}){Style.RESET_ALL}")
        continue

    # 6.5) Any other input → Perform sentiment analysis
    polarity = TextBlob(user_input).sentiment.polarity
    if polarity > 0.25:
        sentiment_type = "Positive"
        color = Fore.GREEN
        emoji = "😊"
    elif polarity < -0.25:
        sentiment_type = "Negative"
        color = Fore.RED
        emoji = "😢"
    else:
        sentiment_type = "Neutral"
        color = Fore.YELLOW
        emoji = "😐"

    # Add the data to conversation history
    conversation_history.append((user_input, polarity, sentiment_type))

    # Print the result of the sentiment analysis to the user
    print(f"{color}{emoji} {sentiment_type} sentiment detected! "
          f"(Polarity: {polarity:.2f}){Style.RESET_ALL}")
