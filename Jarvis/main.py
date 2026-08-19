from brain.model import JarvisBrain
from runtime.jarvis_runtime import JarvisRuntime
from training.coding_learning_demo import run_coding_learning_demo


def main():
    runtime = JarvisRuntime()
    brain = None

    print("\nJARVIS ONLINE")
    print("Commands: /chat, /train, /eval, /status, /learning, /brain, /exit\n")

    while True:
        user_input = input("Raphael > ")
        command = user_input.strip()

        if command in {"/exit", "exit"}:
            break

        if command == "/status":
            print(f"\nJARVIS STATUS > {runtime.status()}\n")
            continue

        if command == "/learning":
            print(f"\nJARVIS LEARNING > {runtime.learning_summary()}\n")
            continue

        if command == "/train":
            print("\nJARVIS > Running controlled CodingWorld learning demo...\n")
            print(run_coding_learning_demo(episodes=6))
            continue

        if command == "/eval":
            print("\nJARVIS > Run /train to execute the before/after CodingWorld benchmark demo.\n")
            continue

        if command.startswith("/brain"):
            message = command.removeprefix("/brain").strip() or input("Brain prompt > ")
            if brain is None:
                brain = JarvisBrain()
                runtime.brain = brain
            print(f"\nJARVIS > {runtime.chat(message)}\n")
            continue

        if command.startswith("/chat"):
            command = command.removeprefix("/chat").strip()

        if brain is None:
            brain = JarvisBrain()
            runtime.brain = brain

        answer = runtime.chat(command)

        print(f"\nJARVIS > {answer}\n")


if __name__ == "__main__":
    main()
