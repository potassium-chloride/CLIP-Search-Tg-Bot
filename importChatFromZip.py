#!/usr/bin/env python3
import utils,zipfile,threading
from PIL import Image

uid2username_cache = dict()

def tgRequest(a,b):
	raise Exception("Not implemented!")

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
	j = [i for i in zp.namelist() if i.endswith('/result.json')]
	if len(j)!=0:
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
	for count,msg in enumerate(j['messages']):
		if not 'photo' in msg:continue
		date = int(msg['date_unixtime'])
		file_id = msg['photo'] # It's not file_id
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
			_ = searchEmbeddingByLink(chat_id,msg_url)
			continue
		except:pass
		try:
			img = Image.open(zp.open(rootdir+msg['photo']))
			json_emb = img2vec(img)
			photo_emb = json_emb['clip']
			if len(text)>23:
				text = text[:10]+"..."+text[-10:]
			try:
				utils.addEmbedding(chat_id, msg_url, file_id, author, embedding=json_emb, msg_text = text, date = date)
			except utils.sqlite3.OperationalError as e:
				ans = tgRequest("getChat",params=[('chat_id',chat_id)])['result']
				print("getChat:",ans)
				utils.addChat(chat_id,ans)
				utils.addEmbedding(chat_id, msg_url, file_id, author, embedding=json_emb, msg_text = text, date = date)
				print('OperationalError:',e)
			print("Successfully added photo")
		except:
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("text", "🙁Не удалось открыть файл "+msg['photo']+"\nПродолжается импорт..." ),
												("parse_mode","html"),
												("disable_web_page_preview",True)
												])
		if count%30==0:
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

def importZipAsync(chat_id,fname,ignoreInconsistance=True):
	import_zip_state[chat_id] = "<i>Начинается импорт...</i>"
	ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
										("text", import_zip_state[chat_id] ),
										("parse_mode","html"),
										("disable_web_page_preview",True)
										])
	t = threading.Thread(target=importZip,args=(chat_id,fname,ignoreInconsistance))
	t.start()
