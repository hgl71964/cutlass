import re
import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--id', type=int, default=0)
    parser.add_argument('-f', type=str, default=None, help='folder with problems')


    args = parser.parse_args()
    return args


def process_path(path: str):
    problems = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")
    
    # Check if it's a directory
    if os.path.isdir(path):
        print(f"Listing files in directory: {path}")
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                problems.append(file_path)

    # Check if it's a file
    elif os.path.isfile(path):
        # print(f"Contents of file: {path}")
        # with open(path, 'r') as file:
        #     print(file.read())
        problems.append(path)

    else:
        raise RuntimeError(f"{path} is not a regular file or directory.")
    
    return problems

def get_gflops(output: str):
    gflops = []

    # Use regular expression to find the GFLOPs values
    gflops_pattern = r'Grouped\s+GFLOPs:\s+(\d+)'
    gflops_matches = re.findall(gflops_pattern, output)

    for match in gflops_matches:
        gflops.append(int(match))

    ### Print the output (stdout)
    print("Standard Output:")
    print(output)
    print(gflops)

    return gflops

def main():
    args = parse_args()

    # problems = ['./build/problems.txt']
    problems = process_path(args.f)
    print(f'problems: {problems}')
    print('=======================')

    for problem in problems:
        # command = ['./build/examples/24_gemm_grouped/streamk_grouped_gemm', '--verbose=true', '--benchmark=./build/problems.txt']

        command = ['./build/examples/24_gemm_grouped/streamk_grouped_gemm', '--verbose=true', f'--benchmark={problem}']

        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            gflops = get_gflops(result.stdout)

            # Print any errors (stderr)
            if result.stderr:
                print("Standard Error:")
                print(result.stderr)
                raise subprocess.CalledProcessError(returncode=result.returncode, cmd=command, stderr=result.stderr)

        except subprocess.CalledProcessError as e:
            # Handle the case where the command returns a non-zero exit code
            print(f"An error occurred while running the command: {e}")
            raise e

if __name__ == '__main__':
    main()