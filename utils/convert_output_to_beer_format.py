import argparse

def main(output_file, beer_file):
  with open(output_file, 'r') as in_f,\
       open(beer_file, 'w') as out_f:
    for line in in_f:
      tokens = line.strip().split()
      phns = ' '.join(tokens[1].split(','))
      out_f.write(f'{tokens[0]} {phns}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--output_file')
    parser.add_argument('--beer_file')
    args = parser.parse_args()
    main(args.output_file, args.beer_file)
