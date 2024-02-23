from django.db import models
from django.utils import timezone

OPT = (
    ('','Select'),
    ('Open','Open'),
    ('Closed', 'Closed'),
)
class Degree(models.Model):
	degree_name = models.CharField(max_length=100)
	def __str__(self):
		return self.degree_name
class Department(models.Model):
	depart = models.CharField(max_length=100)
	def __str__(self):
		return self.depart
class Skillset(models.Model):
	major_skill = models.CharField(max_length=100)
	def __str__(self):
		return self.major_skill
class User_Detail(models.Model):
	name = models.CharField(max_length=30)
	email = models.EmailField(max_length=30)
	phone_number = models.CharField(max_length=30)
	country = models.CharField(max_length=30,null=True,blank=True)
	state = models.CharField(max_length=30,null=True,blank=True)
	city = models.CharField(max_length=30,null=True,blank=True)
	address = models.TextField(max_length=200,null=True,blank=True)
	username = models.CharField(max_length=200,unique=True)
	password = models.CharField(max_length=200)
	degree_id = models.CharField(max_length=200,null=True,blank=True)
	depart_id = models.CharField(max_length=200,null=True,blank=True)
	skill = models.CharField(max_length=20,null=True,blank=True)
	resume = models.FileField('Upload Resume',upload_to='documents/',null=True)
	def __str__(self):
		return self.username
class Admin_Detail(models.Model):
	name = models.CharField(max_length=30)
	email = models.EmailField(max_length=30)
	phone_number = models.CharField(max_length=30)
	username = models.CharField(max_length=200,unique=True)
	password = models.CharField(max_length=200)
	image = models.FileField('Logo Image',upload_to='documents/',null=True)
	def __str__(self):
		return self.username
class Job_Detail(models.Model):
	user_id = models.ForeignKey(Admin_Detail, on_delete=models.CASCADE)
	company_name = models.CharField('Company Name',max_length=100)
	job_title = models.CharField('Job Title',max_length=300)
	description = models.CharField('Job Description',max_length=300)
	position = models.CharField('Position',max_length=300)
	experience = models.CharField('Experience',max_length=300)
	qualification = models.CharField('Qualification',max_length=100)
	salary = models.CharField('Salary',max_length=300)
	status = models.CharField('Job Status',max_length=30,null=True,choices=OPT)
	date = models.DateField(default=timezone.now())
	def __str__(self):
	    return self.company_name
	def publish(self):
	    self.date = timezone.now()
	    self.save()
class Apply_Job(models.Model):
	user_id = models.ForeignKey(User_Detail, on_delete=models.CASCADE)
	job_id = models.ForeignKey(Job_Detail, on_delete=models.CASCADE)
	position = models.CharField(max_length=100)
	salary = models.CharField(max_length=300)
	date = models.DateField()
	availability = models.CharField(max_length=300)
	name = models.CharField(max_length=300)
	passport = models.CharField(max_length=300)
	address = models.TextField(max_length=2000)
	postcode = models.CharField(max_length=10)
	email = models.EmailField(max_length=30)
	phone = models.CharField(max_length=15)
	mobile = models.CharField(max_length=15)
	dob = models.DateField()
	gender = models.CharField(max_length=10)
	q1 = models.CharField(max_length=30)
	i1 = models.CharField(max_length=100)
	m1 = models.CharField(max_length=100)
	g1 = models.CharField(max_length=100)
	y1 = models.CharField(max_length=100)
	q2 = models.CharField(max_length=30,null=True,blank=True)
	i2 = models.CharField(max_length=100,null=True,blank=True)
	m2 = models.CharField(max_length=100,null=True,blank=True)
	g2 = models.CharField(max_length=100,null=True,blank=True)
	y2 = models.CharField(max_length=100,null=True,blank=True)
	q3 = models.CharField(max_length=30,null=True,blank=True)
	i3 = models.CharField(max_length=100,null=True,blank=True)
	m3 = models.CharField(max_length=100,null=True,blank=True)
	g3 = models.CharField(max_length=100,null=True,blank=True)
	y3 = models.CharField(max_length=100,null=True,blank=True)
	q4 = models.CharField(max_length=30,null=True,blank=True)
	i4 = models.CharField(max_length=100,null=True,blank=True)
	m4 = models.CharField(max_length=100,null=True,blank=True)
	g4 = models.CharField(max_length=100,null=True,blank=True)
	y4 = models.CharField(max_length=100,null=True,blank=True)
	q5 = models.CharField(max_length=30,null=True,blank=True)
	i5 = models.CharField(max_length=100,null=True,blank=True)
	m5 = models.CharField(max_length=100,null=True,blank=True)
	g5 = models.CharField(max_length=100,null=True,blank=True)
	y5 = models.CharField(max_length=100,null=True,blank=True)
	company = models.CharField(max_length=100,null=True,blank=True)
	industry = models.CharField(max_length=100,null=True,blank=True)
	position_company = models.CharField(max_length=30,null=True,blank=True)
	fromdate = models.CharField(max_length=100,null=True,blank=True)
	todate = models.CharField(max_length=100,null=True,blank=True)
	level = models.CharField(max_length=100,null=True,blank=True)
	monthly_salary = models.CharField(max_length=100,null=True,blank=True)
	company1 = models.CharField(max_length=100,null=True,blank=True)
	industry1 = models.CharField(max_length=100,null=True,blank=True)
	position_company1 = models.CharField(max_length=30,null=True,blank=True)
	from1 = models.CharField(max_length=100,null=True,blank=True)
	todate1 = models.CharField(max_length=100,null=True,blank=True)
	level1 = models.CharField(max_length=100,null=True,blank=True)
	monthly_salary1 = models.CharField(max_length=100,null=True,blank=True)
	company2 = models.CharField(max_length=100,null=True,blank=True)
	industry2 = models.CharField(max_length=100,null=True,blank=True)
	position_company2 = models.CharField(max_length=30,null=True,blank=True)
	from2 = models.CharField(max_length=100,null=True,blank=True)
	todate2 = models.CharField(max_length=100,null=True,blank=True)
	level2 = models.CharField(max_length=100,null=True,blank=True)
	monthly_salary2 = models.CharField(max_length=100,null=True,blank=True)
	n1 = models.CharField(max_length=100)
	n2 = models.CharField(max_length=100)
	ocu1 = models.CharField(max_length=100)
	ocu2 = models.CharField(max_length=100)
	com1 = models.CharField(max_length=100)
	com2 = models.CharField(max_length=100)
	contact1 = models.CharField(max_length=10)
	contact2 = models.CharField(max_length=10)
	mail1 = models.EmailField(max_length=30)
	mail2 = models.EmailField(max_length=30)
	ref1 = models.CharField(max_length=100)
	ref2 = models.CharField(max_length=100)
	applied_date = models.DateField(default=timezone.now())
	out_come = models.CharField(max_length=100,null=True,blank=True)
	def __str__(self):
	    return self.name
	def publish(self):
	    self.date = timezone.now()
	    self.save()
class Job_Status(models.Model):
	user_id = models.ForeignKey(User_Detail, on_delete=models.CASCADE)
	job_id = models.ForeignKey(Job_Detail, on_delete=models.CASCADE)
	apply_id = models.ForeignKey(Apply_Job, on_delete=models.CASCADE)
	date_recceived = models.DateField()
	hr_officer = models.CharField(max_length=100)
	out_come = models.CharField(max_length=100)
	reason = models.CharField(max_length=1000)
	date = models.DateField()
	def __str__(self):
		return self.hr_officer