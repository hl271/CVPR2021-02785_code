import os

print(os.getcwd())
print(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
print(os.path.join(os.getcwd(), os.pardir))
