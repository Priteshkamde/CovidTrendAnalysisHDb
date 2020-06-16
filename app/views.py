from __future__ import unicode_literals

from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from django.http import Http404
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

from .forms import LoginForm, RegisterForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import Operations,Participants,Transcript,Meeting
import json

import random
import re
from nltk.stem.snowball import SnowballStemmer
import os
import gensim
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
import nltk
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from collections import OrderedDict
import datetime

from django.http import HttpResponse
from django.views.generic import TemplateView

from django.views.generic import View

import hashlib 

import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import os
import random
#REQUIREMENTS.txt
#moviepy
#noisereduce
#soundfile

@login_required
def index(request):
	news = retrieve_top10_news()
	cluster = {}
	content = []
	title = []
	link = []
	for i in news:
		try:
			cluster[i['Cluster']].append({'c':i['Article Content'][:10]+"...",'t':i['Article Title'], 'l':i['Article Link'], 'r':random.randrange(1, 10)})
			print(i['Article Content'][:10]+"...")
		except:
			cluster[i['Cluster']] = []
			cluster[i['Cluster']].append({'c':i['Article Content'][:10]+"...",'t':i['Article Title'], 'l':i['Article Link'], 'r':random.randrange(1, 10)})
	# print(cluster)
	print(cluster[0])
	cluster = {k: v for k, v in sorted(cluster.items(), key=lambda item: item[0])}

	states_covid_data = retrieve_coviddata()
	data = {}	
	c = 0	
	cases = ''
	deaths = ''
	dates = []
	for i in states_covid_data:
		dates.append(i['date'])
		cases += str(i['cases'])+','
		deaths += str(i['deaths'])+','
		try:
			data[c].append({'cases':i['cases'],'date':i['date'],'deaths':i['deaths']})
		except:
			data[c]=[]
			data[c].append({'cases':i['cases'],'date':i['date'],'deaths':i['deaths']})
		c=c+1	
	print("\n",data[0])

	labels = '\"'+'\",\"'.join(dates)+"\""
	
	return render(request, 'website/index2.html', {'cluster':cluster,'covid_data':data, 'deaths':deaths, 'labels':labels, 'cases':cases})

def health(request):
    state = {"status": "UP"}
    return JsonResponse(state)

def handler404(request, exception):
    return render(request, '404.html', status=404)

def handler500(request):
    return render(request, '500.html', status=500)

def checkLogin(request):
	if request.user.is_authenticated:
		return True
	else:
		return False


class RegisterFormView(View):
	form_class = RegisterForm
	template_name = 'website/register.html'
	def get(self, request):
		form = self.form_class(None)
		error, err_email = '', ''
		try:
			error = request.session.get('err_mess', '')
			err_email = request.session.get('err_email', '')
			del request.session['err_email']
			del request.session['err_mess']
		except:
			pass

		return render(request, self.template_name, {'cart_size' : request.session.get('cart_size', 0),'form':form, 'err_email': err_email, 'register_error': error})

	def post(self, request):
		form = self.form_class(request.POST)
		if "already exists" in str(form.errors):
			request.session['err_email'] = request.POST.get('username', '')
			request.session['err_mess'] = "duplicate"
			return redirect('/register/')

		if form.is_valid():
			user = form.save(commit=False)
			username = form.cleaned_data['username']
			password = form.cleaned_data['password']

			acount = 0
			for cr in username:
				if cr == "@":
					acount += 1

			if acount != 1:
				request.session['err_mess'] = "invalidemail"
				return redirect('/register/')

			prevuser = User.objects.filter(email=username)
			if prevuser.count()>0:
				request.session['err_mess'] = "duplicate"
				return redirect('/register/')

			user.set_password(password)
			user.email = username
			user.save()
			user = authenticate(username=username, password=password)

			if user is not None:
				if user.is_active:
					login(request, user)
					lr = request.session.get("lr","index")
					return redirect(lr)
				else:
					return redirect('/login?login=disabled')
			else:
				return redirect('/login?login=failed')

class LoginFormView(View):
	form_class = LoginForm
	template_name = 'website/login.html'
	def get(self, request):
		# 3 -> login to checkout
		request.session['lr'] = request.GET.get('next','index')
		error = request.GET.get('login_error', '')
		if str(request.GET.get('m')) == '3' :
			error = 'login_to_checkout'

		form = self.form_class(None)
		print("Error Value = "+error)
		return render(request, self.template_name, {'cart_size' : request.session.get('cart_size', 0),'form':form, 'login_error': error })


	def post(self, request):
		form = self.form_class(request.POST)
		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(username=username, password=password)
		useremail = User.objects.filter(email=username)

		if user is not None:
			if user.is_active:
				login(request, user)
				lr = request.session.get("lr","index")
				return redirect(lr)
			else:
				return redirect('/login?login_error=disabled')
		else:
			if useremail.count()>0:
				return redirect('/login?login_error=incorrect')
			else:
				return redirect('/login?login_error=failed')

from django.contrib.auth import logout


def logoutForm(request):
	logout(request)
	# Redirect to a success page.
	return redirect('index')

import requests 
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 

import requests
nltk.download('stopwords')
nltk.download('punkt')

def tokenize(train_texts):
	filtered_tokens = []
	tokens = [word for sent in nltk.sent_tokenize(train_texts) for word in nltk.word_tokenize(sent)]
	for token in tokens:
		if re.search('[a-zA-Z]',token):
			if (('http' not in token) and ('@' not in token) and ('<.*?>' not in token) and token.isalnum() and (not token in stop_words)):
				filtered_tokens.append(token)
	return filtered_tokens

def tokenize_stem(train_texts):
	tokens = tokenize(train_texts)
	stemmer = SnowballStemmer('english')
	stemmed_tokens = [stemmer.stem(token) for token in tokens]
	return stemmed_tokens

seed = 137
def load_data(seed):
	train_texts = []
	for index,data in df_news.iterrows():
		train_texts.append(str(data))
	random.seed(seed)
	random.shuffle(train_texts)
	return train_texts

def CovidStatScript(request):
	
	os.system("script.py 1")
	df1 = df.read_csv(os.getcwd()+"/app/static/data/us-states.csv", index_label = 'id')


	request_url = "https://prit-1-priteshkamde24.harperdbcloud.com"

	state_coordinates  = pd.read_csv(os.getcwd()+"/app/static/data/location.csv", index_col="id") 

	url = "https://covid-trend-analysis.herokuapp.com/"
	csv_url = url+"static/data/us-states.csv"
	payload = "{\r\n  \"operation\":\"csv_url_load\",\r\n  \"action\": \"insert\",\r\n  \"schema\":\"cluster\",\r\n  \"table\":\"us_states\",\r\n  \"csv_url\":\""+csv_url+"\"\r\n}"
	headers = {
	    'Content-Type': "application/json",
	    'Authorization': "Basic cHlyYXB0b3JzOmhlbGxvMTIz",
	    'Cache-Control': "no-cache",
	    'Postman-Token': "1ee731fb-761f-427c-9611-35c38138c8fa"
	    }

	response = requests.request("POST", request_url, data=payload, headers=headers)

	request_url = "https://prit-1-priteshkamde24.harperdbcloud.com"
	url = "https://covid-trend-analysis.herokuapp.com/"
	csv_url = url+"static/data/us-location.csv"
	payload = "{\r\n  \"operation\":\"csv_url_load\",\r\n  \"action\": \"insert\",\r\n  \"schema\":\"cluster\",\r\n  \"table\":\"us_location\",\r\n  \"csv_url\":\""+csv_url+"\"\r\n}"
	headers = {
	    'Content-Type': "application/json",
	    'Authorization': "Basic cHlyYXB0b3JzOmhlbGxvMTIz",
	    'Cache-Control': "no-cache",
	    'Postman-Token': "1ee731fb-761f-427c-9611-35c38138c8fa"
	    }

	response = requests.request("POST", request_url, data=payload, headers=headers)


	# result = result.read_csv(os.getcwd()+"/app/static/data/us-location.csv",index_label = 'id')

	# result = pd.merge(df1,state_coordinates,on='state')
	# result.to_csv(os.getcwd()+"/app/static/data/us-location.csv", index_col="id")

	request_url = "https://prit-1-priteshkamde24.harperdbcloud.com"
	url = "https://covid-trend-analysis.herokuapp.com/"
	csv_url = url+"static/data/news_data.csv"
	payload = "{\r\n  \"operation\":\"csv_url_load\",\r\n  \"action\": \"insert\",\r\n  \"schema\":\"cluster\",\r\n  \"table\":\"news_data\",\r\n  \"csv_url\":\""+csv_url+"\"\r\n}"
	headers = {
	    'Content-Type': "application/json",
	    'Authorization': "Basic cHlyYXB0b3JzOmhlbGxvMTIz",
	    'Cache-Control': "no-cache",
	    'Postman-Token': "1ee731fb-761f-427c-9611-35c38138c8fa"
	    }

	response = requests.request("POST", request_url, data=payload, headers=headers)
	return HttpResponse("Started the job")
	
import json
def retrieve_top10_news():
	request_url = "https://prit-1-priteshkamde24.harperdbcloud.com"
	payload = "{\r\n  \"operation\":\"sql\",\r\n  \"sql\": \"SELECT TOP 10 * FROM cluster.news_data\"\r\n}"
	headers = {
    	'Content-Type': "application/json",
    	'Authorization': "Basic cHlyYXB0b3JzOmhlbGxvMTIz"
    }
	response = requests.request("POST", request_url, data=payload, headers=headers)
	data = response.json()
	return data

def retrieve_coviddata():
	request_url = "https://prit-1-priteshkamde24.harperdbcloud.com"
	payload = "{\r\n  \"operation\":\"sql\",\r\n  \"sql\": \"SELECT SUM(cases) as cases,SUM(deaths) as deaths,date FROM cluster.us_states GROUP BY date ORDER BY date ASC\"\r\n}"
	headers = {'Content-Type': "application/json",'Authorization': "Basic cHlyYXB0b3JzOmhlbGxvMTIz"}
	response = requests.request("POST", request_url, data=payload, headers=headers)
	data = response.json()
	print(data)
	return data	 