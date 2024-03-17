#!/home/dolgikh/anaconda3/envs/StableDiffusion/bin/python
import utils,time,torch

test_chat_id = -1

utils.addChat(test_chat_id,["Тестовый"])

url = "https://sun9-79.userapi.com/impf/VLzNdBtJ1m3Ld7PMbUklGRrAAkYdiAjIM26Qaw/NdUPLjv-i4Q.jpg?size=1024x1024&quality=95&sign=8e96646ab3a9508ac059c63664d3b68d&type=album"

emb = utils.img2vec(url)

print("emb['clip'].shape", emb['clip'].shape)
print("emb", emb)

utils.addEmbedding(test_chat_id,url,"file id todo",author = "Dolgikh_KA",embedding=emb)

url = "https://sun9-54.userapi.com/impf/c624320/v624320285/33a33/XhUm7K4aDzg.jpg?size=1279x718&quality=96&sign=7203ac432726cea6c72a4a49da29cfc2&type=album"
emb2 = utils.img2vec(url)
utils.addEmbedding(test_chat_id,url,"file id todo2",author = "Dolgikh_KA",embedding=emb2)

url = "https://sun9-38.userapi.com/impf/71ykXTCurWkBwchuJ2wEexkgaz4BWqdBvwCNGw/LxRDn7sPhAg.jpg?size=590x590&quality=96&sign=3d25f52d5d3d3b070c71d77894cd0819&type=album"
emb3 = utils.img2vec(url)
utils.addEmbedding(test_chat_id,url,"file id todo3",author = "Dolgikh_KA",embedding=emb3)

url = "/tmp/txtmeme.jpg"
emb4 = utils.img2vec(url)
utils.addEmbedding(test_chat_id,url,"file id todo4",author = "Dolgikh_KA",embedding=emb4)

print("Add embeddings successful")

#prompt = "Рисунок Рика Санчез с открытым ртом в костюме Падору на голубом фоне"
#prompt = "Ч/б фото молодого улыбающегося парня в рубашке в сельской местности"
#prompt = "Рисунок рыжего мальчика с канцелярскими принадлежностями в руках"
prompt = "Мем сверхразум с текстом и скриншотами с Reddit"
print("Is prompt English?",utils.isItEnglish(prompt))
prompt_en = utils.ru2en(prompt)

print(prompt,'->',prompt_en)
print("Is prompt English?",utils.isItEnglish(prompt_en))

emb_text = utils.text2vec(prompt_en)

print("Rick-Padoru:",torch.Tensor(emb['clip']).dot(emb_text)/emb_text.norm()/torch.Tensor(emb['clip']).norm())
print("My photo   :",torch.Tensor(emb2['clip']).dot(emb_text)/emb_text.norm()/torch.Tensor(emb2['clip']).norm())
print("My avatar  :",torch.Tensor(emb3['clip']).dot(emb_text)/emb_text.norm()/torch.Tensor(emb3['clip']).norm())
print("Txt meme  :",torch.Tensor(emb4['clip']).dot(emb_text)/emb_text.norm()/torch.Tensor(emb4['clip']).norm())

search_res=utils.searchEmbedding(test_chat_id,emb_text,limit=10)
for i in search_res:print("search_res[i] =",i)
print("***")

search_res=utils.searchEmbedding(test_chat_id,emb_text,limit=10,author = "Dolgikh_KA")
for i in search_res:print("search_res[i] =",i)
print("***")

search_res=utils.searchEmbedding(test_chat_id,emb_text,limit=10,author = "asdfg")
print("search_res =",search_res, "-- Must be void")
print("***")
search_res=utils.searchEmbedding(test_chat_id,emb_text,limit=10,start_date = 0)
for i in search_res:print("search_res[i] =",i)

print("***")
search_res=utils.searchEmbedding(test_chat_id,emb_text,limit=10,end_date = time.time(), needEmbedding=True)
for i in search_res:print("search_res[i] =",i)
