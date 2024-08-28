

#ffconcat version 1.0
#file image01.png
#duration 3
#file image02.png
#duration 5


fd  = open('in.ffconcat', 'w')
fd.write('ffconcat version 1.0\n') 

for i in range(0,900):

	img_name = str(i+1) + '.0.png'
	s = 'file ' + img_name + '\n'
	fd.write(s)
	fd.write('duration 0.1666666\n') 
