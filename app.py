from Generator import gen
from pathlib import Path

def main():
    file_location =  Path(input("Enter the voice file path: ").replace("\"", "").replace("\'", ""))
    sentence = input("Enter the sentence to be cloned: ")
    output_path = gen(file_location, sentence)
    print("Your result is saved on the following path: ", output_path)


if __name__ == '__main__':
    main()