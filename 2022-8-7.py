def create_generator():
    mylist = range(3)
    for i in mylist:
        yield i*i

mygenerator = create_generator()
print(mygenerator)
for i in mygenerator:
    print(i)

for i in mygenerator:
    print(i)