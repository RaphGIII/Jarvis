from brain.model import JarvisBrain


def main():

    brain = JarvisBrain()

    print("\nJARVIS ONLINE")
    print("Type 'exit' to stop.\n")

    while True:

        user_input = input("Raphael > ")

        if user_input.lower() == "exit":
            break

        answer = brain.think(user_input)

        print(f"\nJARVIS > {answer}\n")


if __name__ == "__main__":
    main()