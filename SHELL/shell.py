import subprocess



while True:
    command = input("nocnus> ")
    if command.strip() == "": #if press enter do nothing
        continue
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")

    except Exception as e:
        print(f"nocnus: error: {e}")

        