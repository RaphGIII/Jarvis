from brain.model import JarvisBrain
from runtime.jarvis_runtime import JarvisRuntime
from training.coding_brain_v02_demo import run_coding_brain_v02_demo


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

        if command.startswith("/train"):
            parts = command.split()
            episodes = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 36
            print("\nJARVIS > Running persistent-safe CodingWorld v0.2 training/eval demo...\n")
            print(run_coding_brain_v02_demo(train_episodes=episodes))
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
