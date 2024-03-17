#!/home/dolgikh/anaconda3/envs/StableDiffusion/bin/python
#!/usr/bin/env python3
import utils
import os,sys,time,json,requests,traceback
import tempfile

try:
	from tqdm import tqdm
except:
	tqdm = lambda x:x

def ttt():
	return time.strftime("[%H:%M:%S]: ")

cfg = json.loads(open('config.json','r').read())

tg_api = "https://api.telegram.org/bot"+cfg['tg_token']+"/"
offset = 0
try:
	with open("offset.txt","r") as fl:
		offset = int(fl.read())
except Exception as e:
	print("Cannot load offset:",e)

def tgRequest(m,params=None):
	r = requests.get(tg_api+m,timeout=10+25*m.count("getUpdates"),params=params)
	return r.json()

print(ttt(),tgRequest("getMe"))

loggedImages=[]

def onMessage(upd):
	print(ttt(),upd)
	if 'channel_post' in upd and not 'message' in upd:
		upd['message'] = upd['channel_post']
		upd['message']['from'] = upd['channel_post']['sender_chat']
	if 'edited_message' in upd and not 'message' in upd:
		upd['message'] = upd['edited_message']
	if 'message' not in upd:
		ans = tgRequest("sendMessage",params=[('chat_id',205176061),
											("text", "Неизвестное событие:\n<pre>"+json.dumps(upd)+"</pre>" ),
											("parse_mode","html"),
											("disable_web_page_preview",True),
											("disable_notification",True)
											])
		return True
	if 'username' in upd['message']['from'] and upd['message']['from']['username']==cfg['username']: return True # Не обрабатывать свои сообщения
	chat_id = upd['message']['chat']['id']
	date = upd['message']['date']
	text = ""
	author = str(upd['message']['from']['id'])
	try:
		author = upd['message']['from']['first_name']+' '+upd['message']['from']['last_name']
		if author[0]==' ':author=author[1:]
		if author[-1]==' ':author=author[:-1]
	except:pass
	try:
		author = upd['message']['from']['username']
	except:pass
	if 'caption' in upd['message']:text = upd['message']['caption']
	elif 'text' in upd['message']:text = upd['message']['text']
	if len(text)>0 and text[0]==' ':text=text[1:]
	if len(text)>0 and text[-1]==' ':text=text[:-1]
	reply_to_message_id = upd['message']['message_id']
	photo_emb = None
	if 'new_chat_title' in upd['message']: # Произошло изменение названия чата
		ans = tgRequest("getChat",params=[('chat_id',chat_id)])['result']
		utils.updateChat(chat_id,ans)
	####### It was ELIF
	if 'photo' in upd['message']:
		photo = upd['message']['photo'][-1] # Max size
		file_id = photo['file_id']
		uniq_msg_hash = str(chat_id)+file_id
		if uniq_msg_hash in loggedImages:
			print("ПОВТОРКА!")
		else:
			file_unique_id = photo['file_unique_id']
			# Что ж как запутано всё...
			file_path = tgRequest("getFile?file_id="+file_id)
			file_path = file_path['result']['file_path']
			url = "https://api.telegram.org/file/bot"+cfg['tg_token']+'/'+file_path
			#print(url)
			json_emb = utils.img2vec(url)
			photo_emb = json_emb['clip']
			msg_url = "https://t.me/c/"+str(chat_id).replace("-100","")+"/"+str(upd['message']['message_id'])
			if len(text)>23:
				text = text[:10]+"..."+text[-10:]
			try:
				# addEmbedding(chat_id,msg_link,file_id,author,embedding,msg_text="",date=None)
				utils.addEmbedding(chat_id, msg_url, file_id, author, embedding=json_emb, msg_text = text, date = date)
			except utils.sqlite3.OperationalError as e:
				ans = tgRequest("getChat",params=[('chat_id',chat_id)])['result']
				print("getChat:",ans)
				utils.addChat(chat_id,ans)
				utils.addEmbedding(chat_id, msg_url, file_id, author, embedding=json_emb, msg_text = text, date = date)
				print('OperationalError:',e)
			print("Successfully added photo")
			loggedImages.append(uniq_msg_hash)
			if chat_id>0: # private chat, silent notify
				writing_emoji=json.dumps( [{'type':'emoji','emoji':"✍"}] )
				ans = tgRequest("setMessageReaction",params=[('chat_id',chat_id),
															('message_id',upd['message']['message_id']),
															('reaction', writing_emoji )
															])
	#######
	if text.lower().startswith('/start'): # Tests
		ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
											("reply_to_message_id",reply_to_message_id),
											("text", "Hello!\nThis bot allows your personal image search in your chats. This bot <b>DO NOT SEARCH VIDEOS WITH HOT GIRLS ON GLOBAL INTERNET</b>. Этот бот <b>не ищет видео с горячими девушками в глобальном Интернете</b>. Please, call /help for more details." ),
											("parse_mode","html"),
											("disable_web_page_preview",True)
											])
		#print(tgRequest("sendMessage?chat_id="+str(chat_id)+"&reply_to_message_id="+str(reply_to_message_id)+"&text=Hello"))
		ans = tgRequest("sendMessage",params=[('chat_id',205176061),
											("text", "Новый пользователь:\n<pre>"+json.dumps(upd)+"</pre>" ),
											("parse_mode","html"),
											("disable_web_page_preview",True),
											("disable_notification",True)
											])
	elif text.lower().startswith('ping'): # Tests
		ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
											("reply_to_message_id",reply_to_message_id),
											("text", "pong! dt="+str(round(time.time()-date,3))+"s" ),
											("parse_mode","html"),
											("disable_web_page_preview",True)
											])
	elif text.lower().startswith('/help'): # Help
		helptxt = open("help.txt","r").read()
		ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
											("reply_to_message_id",reply_to_message_id),
											("text", helptxt ),
											("parse_mode","html"),
											("disable_web_page_preview",True)
											])
		print(ans)
	elif text.lower().startswith('/setcontext'):
		if chat_id<0:
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("reply_to_message_id",reply_to_message_id),
												("text", "<i>Эта функция только для личных чатов</i>" ),
												("parse_mode","html")
												])
			return True
		chatName = text[len(text.split(' ')[0])+1:]
		if len(chatName)<5:
			settings = utils.getChat(chat_id)
			settings['context'] = chat_id
			utils.updateChat(chat_id,settings)
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("reply_to_message_id",reply_to_message_id),
												("text", "<i>Контекст сброшен на личный</i>" ),
												("parse_mode","html")
												])
			return True
		availChats = utils.findChatByName(chat_id,chatName)
		if len(availChats)==0:
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("reply_to_message_id",reply_to_message_id),
												("text", "<i>Чата с таким названием не найдено :(</i>" ),
												("parse_mode","html")
												])
			return True
		if len(availChats)>1:
			ans = "Вот, что мне удалось найти:\n"+'\n'.join([i['title'] for i in j])
			ans += "\nНаберите чат так, чтобы он нашёлся только один"
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("reply_to_message_id",reply_to_message_id),
												("text", ans ),
												("parse_mode","html")
												])
			return True
		context_id = availChats[0]['id']
		settings = utils.getChat(chat_id)
		settings['context'] = context_id
		utils.updateChat(chat_id,settings)
		ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
											("reply_to_message_id",reply_to_message_id),
											("text", "Теперь поиск в этом личном чате будет выполняться как в чате "+availChats[0]['title'] ),
											("parse_mode","html")
											])
		return True
		#print(ans)
	elif text.lower().startswith('/status'): # Help
		status = "<i>Нет данных</i>"
		if chat_id in utils.import_zip_state:
			status = utils.import_zip_state[chat_id]
		ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
											("reply_to_message_id",reply_to_message_id),
											("text", status ),
											("parse_mode","html"),
											("disable_web_page_preview",True)
											])
		#print(ans)
	elif text.lower().startswith('/import'): # Help
		if chat_id in utils.import_zip_state:
			status = utils.import_zip_state[chat_id]
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("reply_to_message_id",reply_to_message_id),
												("text", status ),
												("parse_mode","html"),
												("disable_web_page_preview",True)
												])
			return True
		if not 'document' in upd['message'] or upd['message']['document']['mime_type']!='application/zip':
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("reply_to_message_id",reply_to_message_id),
												("text", "Скачайте историю чата в машиночитаемом формате JSON вместе с фотографиями, запакуйте в zip-архив и прикрепите к команде import" ),
												("parse_mode","html"),
												("disable_web_page_preview",True)
												])
			#print(ans)
			return True
		file_id = upd['message']['document']['file_id']
		file_path = tgRequest("getFile?file_id="+file_id)
		if not file_path['ok']:
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
												("reply_to_message_id",reply_to_message_id),
												("text", "Ошибка загрузки: "+file_path['description']+"\nSee more: https://core.telegram.org/bots/api#getfile" ),
												("parse_mode","html"),
												("disable_web_page_preview",True)
												])
			return True
		file_path = file_path['result']['file_path']
		url = "https://api.telegram.org/file/bot"+cfg['tg_token']+'/'+file_path
		fl = tempfile.TemporaryFile()
		with requests.get(url, stream=True) as r:
			r.raise_for_status()
			for chunk in r.iter_content(chunk_size=8192):
				fl.write(chunk)
		utils.importZipAsync(chat_id,fl,ignoreInconsistance=False)
	elif text.lower().startswith('/найди') or text.lower().startswith('/find'): # Help
		msg_url = None # Var for replies
		text,inntext = utils.getStrictText(text)
		text,start_date,end_date = utils.getTimestampsFromPrompt(text)
		text = text.replace("  "," ").replace("  "," ").replace("  "," ")
		wordarr=text.lower().split(" ")
		text = text[len(wordarr[0])+1:]
		limit = 5
		author = None
		if len(wordarr)>1 and wordarr[1] in ['топ','top', 'limit']:
			try:
				limit = int(wordarr[2])
				if limit>100:
					limit = 100
				text = text[len(wordarr[1])+1:]
				text = text[len(wordarr[2])+1:]
				print("limit =",limit)
			except:pass
		if chat_id<0 and len(wordarr)>1 and wordarr[-2] in ['от','from']:
			try:
				author = wordarr[-1]
				if author[0]=="@": author=author[1:]
				text = text[:-len(wordarr[-1])-1]
				text = text[:-len(wordarr[-2])-1]
				print("author =","'"+author+"'")
			except:pass
		elif chat_id<0 and len(wordarr)>2 and wordarr[-3] in ['от','from']:
			try:
				author = wordarr[-2]+" "+wordarr[-1]
				text = text[:-len(wordarr[-1])-1]
				text = text[:-len(wordarr[-2])-1]
				text = text[:-len(wordarr[-3])-1]
				print("author =","'"+author+"'")
			except:pass
		if author is not None and author[0]==' ':author=author[1:]
		if author is not None and author[-1]==' ':author=author[:-1]
		prompt = text
		print("prompt =",prompt)
		if (len(prompt)<5 or prompt.count(" ")==0) and photo_emb is None:
			if 'reply_to_message' in upd['message']:
				msg_id = upd['message']['reply_to_message']['message_id']
				msg_url = "https://t.me/c/"+str(chat_id).replace("-100","")+"/"+str(msg_id)
				photo_emb = utils.searchEmbeddingByLink(chat_id, msg_url)
				if type(photo_emb)==list and len(photo_emb)==0:
					ans = tgRequest("sendMessage",params=[('chat_id',chat_id),("reply_to_message_id",reply_to_message_id),
														("text", "<i>Невозможно выполнить запрос :(</i>" ),
														("parse_mode","html")])
					return True
		if (len(prompt)<5 or prompt.count(" ")==0) and photo_emb is not None:
			emb_text = photo_emb
		else:
			if not utils.isItEnglish(prompt):
				prompt = utils.ru2en(prompt)
			emb_text = utils.text2vec(prompt)
		context_chat_id = chat_id
		if chat_id>0: # Check local context settings
			settings = utils.getChat(chat_id)
			if 'context' in settings:
				# Check is in group
				isMember = settings['context']>0 or tgRequest('getChatMember',[('chat_id',settings['context']),('user_id',chat_id)])['ok']
				if not isMember:
					settings['context'] = chat_id
					utils.updateChat(chat_id,settings)
					ans = tgRequest("sendMessage",params=[('chat_id',chat_id),("reply_to_message_id",reply_to_message_id),
														("text", "<i>Вы больше не состоите в чате, ваш контекст поиска сброшен на личный</i>" ),
														("parse_mode","html")])
				else:
					context_chat_id = settings['context']
		try:
			print("start_date =",start_date)
			print("end_date =",end_date)
			if emb_text is None and author is None and start_date>=end_date and inntext is None:
				ans = tgRequest("sendMessage",params=[('chat_id',chat_id),("reply_to_message_id",reply_to_message_id),
													("text", "<i>Задан пустой запрос. Передайте текстовый запрос, фильтры поиска и/или изображение</i>" ),
													("parse_mode","html")])
				print(ans)
				return True
			search_res=utils.searchEmbedding(context_chat_id,emb_text, author=author, limit=limit, start_date = start_date, end_date = end_date, inntext = inntext)
			# searchEmbedding(chat_id, embedding, author=None, limit=10, start_date = 0, end_date = -1, needEmbedding = False):
		except utils.sqlite3.OperationalError as e:
			ans = tgRequest("sendMessage",params=[('chat_id',chat_id),("reply_to_message_id",reply_to_message_id),
												("text", "<i>Произошла непридвиденная ошибка, возможно, вы ещё не добавили ни одной фотографии :(</i>" ),
												("parse_mode","html")])
			print(ans)
			return True
		res = "Вот что мне удалось найти:\n"
		counter = 1
		for img in search_res:
			if msg_url is not None and img['msg_text'] == msg_url:
				continue # Это реплай на своё же, игнорим
			msg_text = "<i>...</i>"
			if len(img['msg_text'])>0:
				msg_text = img['msg_text']
			res+=str(counter)+") "
			res+="<a href='"+img['msg_link']+"'>"
			res+=utils.time2str(img['date'])+" \""
			res+=msg_text
			res+="\" от "+img['author']
			res+="</a>"
			#res+=" - ≈"+str(round(100*img['score']))+"%"
			res+="\n"
			counter += 1
		if len(search_res)==0:
			res ="<i>Ничего не найдено :(</i>"
		res = res.replace(cfg['tg_token'],"...")
		ans = tgRequest("sendMessage",params=[('chat_id',chat_id),
											("reply_to_message_id",reply_to_message_id),
											("text", res ),
											("parse_mode","html"),
											("disable_web_page_preview",True)
											])
		if len(search_res)>0: # Сделаем бота более осмысленным в личке
			media_arr = res.split("\n")[1:]
			media_arr = [{'type':'photo',
						'media':sres['file_id'],
						'caption':desc,
						'parse_mode':'html'} for desc,sres in zip(media_arr, search_res) if not sres['file_id'].lower().endswith('.jpg')]
			ans = tgRequest("sendMediaGroup",params=[('chat_id',chat_id),('media',json.dumps(media_arr))])
		print(ans)
	return True


while True:
	try:
		offset_changed = False
		updates = tgRequest("getUpdates?timeout=25&offset="+str(offset))
		if updates['ok']:
			updates = updates['result']
			for u in updates:
				if onMessage(u):
					offset = u['update_id']+1
					offset_changed = True
		else:
			print(updates)
		if offset_changed:
			with open("offset.txt","w") as fl:
				fl.write(str(offset))
	except KeyboardInterrupt:
		exit(0)
	except Exception as e:
		print("updates =",updates)
		print(e,traceback.format_exc())
		time.sleep(1)
		print("Continue!")
#		raise e
