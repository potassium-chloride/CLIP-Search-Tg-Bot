#!/usr/bin/env python3
import requests,json
import sqlite3,sys,time
import pytesseract
import threading,zipfile

cfg = json.loads(open('config.json','r').read())

tg_api = "https://api.telegram.org/bot"+cfg['tg_token']+"/"

def tgRequest(m,params=None):
	r = requests.get(tg_api+m,timeout=10+25*m.count("getUpdates"),params=params)
	return r.json()

uid2username_cache = dict()

def uid2username(uid,chat_id=None):
	global uid2username_cache
	if type(uid)==str and uid.startswith("user"):
		uid = int(uid[4:])
	elif type(uid)==str:
		uid = int(uid)
	if uid in uid2username_cache:
		return uid2username_cache[uid]
	else:
		try:
			r = tgRequest('getChatMember',[('chat_id',chat_id),('user_id',uid)])
			r = r['result']['user']
			if 'username' in r:
				name = r['username']
			else:
				name = r['first_name']
				try: name += " "+r['last_name']
				except:pass
			uid2username_cache[uid] = name
			return name
		except:pass
		try:
			r = tgRequest('getChat',[('chat_id',uid)])
			r = r['result']
			if 'username' in r:
				name = r['username']
			else:
				name = r['first_name']
				try: name += " "+r['last_name']
				except:pass
			uid2username_cache[uid] = name
			return name
		except:pass
		name = "user"+str(uid)
		uid2username_cache[uid] = name
		return name

import_zip_state=dict()

def importZip(chat_id,fname,ignoreInconsistance=True):
	zp = zipfile.ZipFile(fname,'r')
	j = [i for i in zp.namelist() if i.endswith('result.json')]
	print(j)
	if len(j)!=1:
		import_zip_state[chat_id] = "<i>Импорт завершился неудачно. Проверьте, что вы выбрали машиночитаемый формат JSON и в архиве есть файл </i>result.json"
		ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
											("text", import_zip_state[chat_id] ),
											("parse_mode","html"),
											("disable_web_page_preview",True)
											])
		return
	j = j[0]
	print("j =",j)
	rootdir = j.replace('result.json','')
	print("rootdir =",rootdir)
	j = zp.open(j).read()
	j = json.loads(j)
	start_t = time.time()
	if not ignoreInconsistance:
		if str(chat_id)!="-100"+str(j['id']):
			import_zip_state[chat_id] = "<i>Импорт прерван, так как id данного чата не совпадает с id в архиве для импорта.</i>"
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("text", import_zip_state[chat_id] ),
												("parse_mode","html"),
												("disable_web_page_preview",True)
												])
			return
	for count,msg in enumerate(j['messages']):
		if not 'photo' in msg:continue
		date = int(msg['date_unixtime'])
		file_id = msg['photo'] # It's not file_id!
		msg_id = msg['id']
		text = msg['text']
		msg_url = "https://t.me/c/"+str(j['id'])+"/"+str(msg_id)
		author = uid2username(msg['from_id'],chat_id=chat_id)
		if author==msg['from_id']:
			try:
				if len(msg['from'])>1:
					author = msg['from']
			except:pass
		try:
			_ = searchEmbeddingByLink(chat_id,msg_url) # Already exist
			if len(_)>0:continue
		except:pass
		try:
			if msg['photo'][-4:].lower() not in [".jpg","jpeg",".png"]:
				continue
			img = Image.open(zp.open(rootdir+msg['photo']))
			json_emb = img2vec(img)
			photo_emb = json_emb['clip']
			if len(text)>23:
				text = text[:10]+"..."+text[-10:]
			try:
				addEmbedding(chat_id, msg_url, file_id, author, embedding=json_emb, msg_text = text, date = date)
			except sqlite3.OperationalError as e:
				ans = tgRequest("getChat",params=[('chat_id',chat_id)])['result']
				print("getChat:",ans)
				addChat(chat_id,ans)
				addEmbedding(chat_id, msg_url, file_id, author, embedding=json_emb, msg_text = text, date = date)
				print('OperationalError:',e)
			print("Successfully added photo")
		except Exception as e:
			print(e)
			print(rootdir+msg['photo'])
			print(chat_id, msg_url, file_id, author, json_emb, text, date = date)
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("text", "🙁Не удалось проиндексировать файл "+msg['photo']+"\nПродолжается импорт..." ),
												("parse_mode","html"),
												("disable_web_page_preview",True)
												])
		if count==1 or count%30==0:
			spent_time = time.time()-start_t
			percent = 100*(count+1)/len(j['messages'])
			state = "<i>Импорт продолжается, завершено "+str(round(percent))+"\n"
			speed = spent_time/(count+1)
			wait_time = (len(j['messages'])-count)*speed
			state +="Осталось примерно "+str(round(wait_time/60))+" минут</i>"
			import_zip_state[chat_id] = state
	import_zip_state[chat_id] = "✅Импорт завершён!"
	ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
										("text", import_zip_state[chat_id] ),
										("parse_mode","html"),
										("disable_web_page_preview",True)
										])
	if not type(fl)==str:
		fl.close()

def importZipAsync(chat_id,fname,ignoreInconsistance=True):
	if chat_id in import_zip_state:
		ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
											("text", import_zip_state[chat_id] ),
											("parse_mode","html"),
											("disable_web_page_preview",True)
											])
		return False
	import_zip_state[chat_id] = "<i>Начинается импорт...\nИспользуйте команду /status, чтобы узнать прогресс</i>"
	ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
										("text", import_zip_state[chat_id] ),
										("parse_mode","html"),
										("disable_web_page_preview",True)
										])
	t = threading.Thread(target=importZip,args=(chat_id,fname,ignoreInconsistance),daemon = True)
	import_zip_state[chat_id] = "<i>Начинается импорт...</i>"
	t.start()
	return True

########## SQL features

#conn = sqlite3.connect(cfg['database'])
#cursor = conn.cursor()

def initDB():
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	cursor.execute("create table chats (id INTEGER,json VARCHAR, PRIMARY KEY (id));")
	print(cursor.fetchall())
	conn.commit()
	conn.close()

try:
	print("Init DB")
	initDB()
except Exception as e:
	print("Init DB result:",e)

def addChat(uid,j):
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	cursor.execute("insert into chats (id,json) values (?,?)",(uid,json.dumps(j)))
	compiled=""
	for i in range(768):
		compiled += ", clip"+str(i)+" REAL"
	cursor.execute("create table chat"+str(uid).replace("-","_")+" (msg_link VARCHAR, date INTEGER, file_id VARCHAR, author VARCHAR, embedding VARCHAR, msg_text VARCHAR"+compiled+");")
	conn.commit()
	conn.close()
	return 0#cursor.fetchall()

def updateChat(uid,j):
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	cursor.execute("update chats set json=? where id=?",(json.dumps(j),uid))
	conn.commit()
	conn.close()
	return 0#cursor.fetchall()

def getChat(uid):
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	cursor.execute("select json from chats where id=?",(uid,))
	try:
		j = json.loads(cursor.fetchall()[0][0])
	except:
		j = {}
	conn.close()
	return j#cursor.fetchall()

def findChatByName(uid,name):
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	cursor.execute("select json from chats where json like ?",('%'+json.dumps(name)[1:-1]+'%',))
	try:
		j = [json.loads(i[0]) for i in cursor.fetchall()]
		j = j[:10]
		j = [i for i in j if tgRequest('getChatMember',[('chat_id',i['id']),('user_id',uid)])['ok']]
	except:
		j = []
	conn.close()
	return j#cursor.fetchall()

def findImageByText(chat_id,txt,author=None, limit=10, start_date = 0, end_date = -1, needEmbedding = False):
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	if limit<1:
		return []
	query = "select msg_link,date,file_id,author,embedding,msg_text from chat"+str(chat_id).replace("-","_")
	query +=" where embedding like ?"
	if author is not None:
		query += " and lower(author) = ?"
	if start_date>0:
		query +=" and date>"+str(start_date)
	if end_date>0:
		query +=" and date<"+str(end_date)
	query +=" limit "+str(limit)
	if author is None:
		cursor.execute(query,('%'+json.dumps(txt)[1:-1]+'%',))
	else:
		cursor.execute(query,('%'+json.dumps(txt)[1:-1]+'%',author.lower()))
	res = cursor.fetchall()
	conn.close()
	# Post process results
	if len(res)==0:return []
	j = []
	embs = []
	for i,obj in enumerate(res):
		msg_link,date,file_id,author,embeding,msg_text = obj
		json_emb = json.loads(embeding)
		embs.append(json_emb['clip'])
		j.append({'msg_link':msg_link,
					'date':date,
					'msg_text':msg_text,
					'file_id':file_id,
					'author':author})
		if needEmbedding:
			j[-1]['embeddings'] = json_emb
			j[-1]['embeddings']['clip'] = torch.Tensor(j[-1]['embeddings']['clip'])
	return j

def addEmbedding(chat_id,msg_link,file_id,author,embedding,msg_text="",date=None):
	if type(msg_text)==list:
		for i in range(len(msg_text)):
			if type(msg_text[i])==dict:
				msg_text[i] = msg_text[i]['text']
		msg_text = "".join(msg_text)
		if len(msg_text)>23:
			msg_text = msg_text[:10]+"..."+msg_text[-10:]
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	if date is None:
		date=round(time.time())
	if type(embedding['clip'])==list:
		embedding['clip'] = torch.Tensor(embedding['clip'])
	lnorm = (embedding['clip']/embedding['clip'].norm()).tolist()
	embedding['clip'] = embedding['clip'].tolist()
	compiled=""
	for i in range(768):
		compiled += ", clip"+str(i)
	query = "insert into chat"+str(chat_id).replace("-","_")
	query += " (msg_link,date,file_id,author,embedding,msg_text"+compiled+") values (?,?,?,?,?,?"
	query += ",?"*768+")"
	values = (msg_link,date,file_id,author,json.dumps(embedding),msg_text) + tuple(lnorm)
#	print(values)
	cursor.execute(query,values)
#	print("OK")
	conn.commit()
	conn.close()
	return 0#cursor.fetchall()

def searchEmbedding(chat_id, embedding, author=None, limit=10, start_date = 0, end_date = -1, inntext=None, needEmbedding = False):
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	if limit<1:
		return []
	if embedding is None:
		print("embedding is None")
		return searchUnconditionalEmbedding(chat_id, author=author, limit=limit, start_date = start_date, end_date = end_date, inntext=None , needEmbedding = needEmbedding)
	if type(embedding)==list:
		embedding = torch.Tensor(embedding)
	embedding = embedding/embedding.norm()
	l = embedding.tolist()
	query = "select msg_link,date,file_id,author,embedding,msg_text from chat"+str(chat_id).replace("-","_")
	q_args = ()
	if author is not None:
		query += " where lower(author) = ?"
		q_args = (author.lower(),)
	if start_date>0:
		if 'where' in query: query +=" and date>"+str(start_date)
		else: query +=" where date>"+str(start_date)
	if end_date>0:
		if 'where' in query: query +=" and date<"+str(end_date)
		else: query +=" where date<"+str(end_date)
	if inntext is not None:
		if 'where' in query: query +=" and embedding like ?"
		else: query +=" where embedding like ?"
		q_args += ('%'+json.dumps(inntext)[1:-1]+'%',)
	query +=" order by ("
	compiled=""
	for i in range(768):
		if l[i]>0 and i>0:
			compiled +="+"
		compiled += str(l[i])+"*clip"+str(i)
	query += compiled
	query +=") desc limit "+str(limit)
	# print(query)
	if author is None and inntext is None:
		cursor.execute(query)
	else:
		cursor.execute(query,q_args)
	res = cursor.fetchall()
	conn.close()
	# Post process results
	if len(res)==0:return []
	j = []
	embs = []
	for i,obj in enumerate(res):
		msg_link,date,file_id,author,embeding,msg_text = obj
		json_emb = json.loads(embeding)
		embs.append(json_emb['clip'])
		j.append({'msg_link':msg_link,
					'date':date,
					'msg_text':msg_text,
					'file_id':file_id,
					'author':author})
		if needEmbedding:
			j[-1]['embeddings'] = json_emb
			j[-1]['embeddings']['clip'] = torch.Tensor(j[-1]['embeddings']['clip'])
	embs = torch.tensor(embs)
	scores = embs.matmul(embedding)/embs.norm(dim=1)
	if len(res)>0:
		scores/= scores.max()
		scores = scores.softmax(dim=0)
	for i in range(len(res)):
		j[i]['score'] = scores[i].item()
	return j

def searchUnconditionalEmbedding(chat_id, author=None, limit=10, start_date = 0, end_date = -1, inntext=None, needEmbedding = False):
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	if limit<1:
		return []
	query = "select msg_link,date,file_id,author,embedding,msg_text from chat"+str(chat_id).replace("-","_")
	q_args = ()
	if author is not None:
		query += " where lower(author) = ?"
		q_args = (author.lower(),)
	if start_date>0:
		if 'where' in query: query +=" and date>"+str(start_date)
		else: query +=" where date>"+str(start_date)
	if end_date>0:
		if 'where' in query: query +=" and date<"+str(end_date)
		else: query +=" where date<"+str(end_date)
	if inntext is not None:
		if 'where' in query: query +=" and embedding like ?"
		else: query +=" where embedding like ?"
		q_args += ('%'+json.dumps(inntext)[1:-1]+'%',)
	query +=" limit "+str(limit)
	# print(query)
	if author is None and inntext is None:
		cursor.execute(query)
	else:
		cursor.execute(query,q_args)
	res = cursor.fetchall()
	conn.close()
	# Post process results
	if len(res)==0:
		print(query)
		return []
	j = []
	for i,obj in enumerate(res):
		msg_link,date,file_id,author,embeding,msg_text = obj
		json_emb = json.loads(embeding)
		j.append({'msg_link':msg_link,
					'date':date,
					'msg_text':msg_text,
					'file_id':file_id,
					'author':author})
		if needEmbedding:
			j[-1]['embeddings'] = json_emb
			j[-1]['embeddings']['clip'] = torch.Tensor(j[-1]['embeddings']['clip'])
	return j

def searchEmbeddingByLink(chat_id, link):
	conn = sqlite3.connect(cfg['database'])
	cursor = conn.cursor()
	query = "select embedding from chat"+str(chat_id).replace("-","_")+" where msg_link = ?"
	cursor.execute(query,(link,))
	res = cursor.fetchall()
	conn.close()
	# Post process results
	if len(res)==0:return []
	embeding = res[0]
	if type(embeding)!=str:
		print("=======================> Всё-таки надо ещё раз: type =",type(embeding))
		embeding = embeding[0]
	json_emb = json.loads(embeding)
	return torch.Tensor(json_emb['clip'])

# AI utils

from PIL import Image
import requests,torch
import numpy as np
from insightface.app import FaceAnalysis
FaceApp = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
FaceApp.prepare(ctx_id=0, det_size=(640, 640))

def PIL2faces(img):
	img = np.array(img)
	img = np.ascontiguousarray(img[...,[2,1,0]])
	faces = FaceApp.get(img)
	faces = [i.normed_embedding.tolist() for i in faces]
	return faces

from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-large-patch14"
#model_name = "openai/clip-vit-base-patch32"

print('Load CLIP model...',file=sys.stderr)
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

def img2vec(img):
	if type(img)==str:
		if img.startswith('http'):
			img = Image.open(requests.get(img, stream=True).raw)
		else:
			img = Image.open(img)
	if max(img.size)>1280: # So big
		f = max(img.size)/1280
		w,h = round(img.size[0]/f),round(img.size[1]/f)
		img = img.resize((w,h))
	res = dict()
	res['ocr_rus'] = pytesseract.image_to_string(img,lang='rus')
	res['ocr_eng'] = pytesseract.image_to_string(img,lang='eng')
#	try:
#		res['faces'] = PIL2faces(img)
#	except Exception as e:
#		print(e,res)
	with torch.no_grad():
		inputs = clip_processor(images=img, return_tensors="pt")
		res['clip'] = clip_model.get_image_features(**inputs)[0]
	return res

def text2vec(txt):
	if len(txt)<2: # Пустой запрос
		print("Пустой промпт")
		return None
	with torch.no_grad():
		inputs = clip_processor(text=txt,padding=True, return_tensors="pt")
		return clip_model.get_text_features(**inputs)[0]

print("Load translator model...")

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "facebook/wmt19-ru-en"
ruen_tokenizer = FSMTTokenizer.from_pretrained(mname)
ruen_model = FSMTForConditionalGeneration.from_pretrained(mname)

def ru2en(txt):
	input_ids = ruen_tokenizer.encode(txt.replace("ё","е").replace("Ё","Е"), return_tensors="pt")
	outputs = ruen_model.generate(input_ids)
	decoded = ruen_tokenizer.decode(outputs[0], skip_special_tokens=True)
	if len(decoded.replace(" ",""))==0 and len(txt.replace(" ",""))>0:
		tgRequest("sendMessage",params=[('chat_id',205176061),
											("text", "Не удалось выполнить перевод промпта:\n<pre>"+txt+"</pre>" ),
											("parse_mode","html"),
											("disable_web_page_preview",True),
											("disable_notification",True)
											])
		return txt
	return decoded

# Other

import datetime
import re

def checkTimeFromPrompt(txt,txtl,test):
	if test not in txtl:
		return False,txt
	i=txtl.index(test)
	return True,txt[:i]+txt[i+len(test):]

def getTimestampsFromPrompt(txt):
	txtl = txt.lower()
	b,txt=checkTimeFromPrompt(txt,txtl,"за последние сутки")
	if b: return txt,round(time.time()-86400),-1
	b,txt=checkTimeFromPrompt(txt,txtl,"за последний час")
	if b: return txt,round(time.time()-3600),-1
	b,txt=checkTimeFromPrompt(txt,txtl,"за последнюю неделю")
	if b: return txt,round(time.time()-86400*7),-1
	reg = re.compile("с \d\d\d\d\.\d\d\.\d\d")
	start = reg.findall(txtl)
	if len(start)>0:
		try:
			start = time.mktime(time.strptime(start[-1][2:],"%Y.%m.%d"))
			txt = txt.replace(start[-1],"").replace("  "," ")
		except:
			start = 0
	else:
		start = 0
	reg = re.compile("по \d\d\d\d\.\d\d\.\d\d")
	end = reg.findall(txtl)
	if len(end)>0:
		try:
			end = time.mktime(time.strptime(end[-1][3:],"%Y.%m.%d"))
			txt = txt.replace(end[-1],"").replace("  "," ")
		except:
			end = -1
	else:
		end = -1
	reg = re.compile("с \d\d\.\d\d\.\d\d\d\d")
	start = reg.findall(txtl)
	if len(start)>0:
		try:
			start = time.mktime(time.strptime(start[-1][2:],"%d.%m.%Y"))
			txt = txt.replace(start[-1],"").replace("  "," ")
		except:
			start = 0
	else:
		start = 0
	reg = re.compile("по \d\d\.\d\d\.\d\d\d\d")
	end = reg.findall(txtl)
	if len(end)>0:
		try:
			end = time.mktime(time.strptime(end[-1][3:],"%d.%m.%Y"))
			txt = txt.replace(end[-1],"").replace("  "," ")
		except:
			end = -1
	else:
		end = -1
	return txt,round(start),round(end)

def getStrictText(txt):
	if "с текстом \"" in txt.lower() and txt.count("\"")>1:
		txt2 = txt[:txt.lower().index("с текстом \"")]+txt[txt.rindex("\"")+1:]
		txt2 = txt2.replace("  "," ")
		inntext = txt[txt.lower().index("с текстом \"")+len("с текстом \""):txt.rindex("\"")]
		inntext = inntext.replace("  "," ")
		if inntext[0]==" ":inntext=inntext[1:]
		if inntext[-1]==" ":inntext=inntext[:-1]
		if txt2[0]==" ":txt2=txt2[1:]
		if txt2[-1]==" ":txt2=txt2[:-1]
		return txt2, inntext
	elif " с текстом " in txt.lower():
		txt2 = txt[:txt.lower().index(" с текстом ")]
		txt2 = txt2.replace("  "," ")
		inntext = txt[txt.lower().index(" с текстом ")+len(" с текстом "):]
		inntext = inntext.replace("  "," ")
		if inntext[0]==" ":inntext=inntext[1:]
		if inntext[-1]==" ":inntext=inntext[:-1]
		if txt2[0]==" ":txt2=txt2[1:]
		if txt2[-1]==" ":txt2=txt2[:-1]
		return txt2, inntext
	elif "with text \"" in txt.lower() and txt.count("\"")>1:
		txt2 = txt[:txt.lower().index("with text \"")]+txt[txt.rindex("\"")+1:]
		txt2 = txt2.replace("  "," ")
		inntext = txt[txt.lower().index("with text \"")+len("with text \""):txt.rindex("\"")]
		inntext = inntext.replace("  "," ")
		if inntext[0]==" ":inntext=inntext[1:]
		if inntext[-1]==" ":inntext=inntext[:-1]
		if txt2[0]==" ":txt2=txt2[1:]
		if txt2[-1]==" ":txt2=txt2[:-1]
		return txt2, inntext
	elif " with text " in txt.lower():
		txt2 = txt[:txt.lower().index(" with text ")]
		txt2 = txt2.replace("  "," ")
		inntext = txt[txt.lower().index(" with text ")+len(" with text "):]
		inntext = inntext.replace("  "," ")
		if inntext[0]==" ":inntext=inntext[1:]
		if inntext[-1]==" ":inntext=inntext[:-1]
		if txt2[0]==" ":txt2=txt2[1:]
		if txt2[-1]==" ":txt2=txt2[:-1]
		return txt2, inntext
	return txt,None

def time2str(date):
	return datetime.datetime.fromtimestamp(date).strftime("%Y.%m.%d %H:%M:%S")

enAlph = 'qwertyuiopasdfghjklzxcvbnm'
ruAlph = 'йцукенгшщзхъфывапролджэячсмитьбю'
def isItEnglish(txt):
	txtl = txt.lower()
	ruScore = sum([i in txtl for i in ruAlph])
	enScore = sum([i in txtl for i in enAlph])
	return enScore>=ruScore

print("Utils loaded!")
