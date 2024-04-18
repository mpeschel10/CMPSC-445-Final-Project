# Source: https://www.leewayhertz.com/build-a-gpt-model/
# 50 Victorian novelists dataset: https://archive.ics.uci.edu/dataset/454/victorian+era+authorship+attribution
# Master's thesis about the 50 Victorian novelists: https://scholarworks.iupui.edu/server/api/core/bitstreams/708a9870-915e-4d59-b54d-938af563c196/content
# Github for the 50 Victorian novelists: https://github.com/agungor2/Authorship_Attribution/

# You can see the code he describes as converting from words to numbers here: https://github.com/agungor2/Authorship_Attribution/blob/master/convert_data.py
# Specifically, he loads a "Vocabulary_wstopwords.mat" file, then accesses the "shortened_vocab" member of it.
# I don't know where the .mat files are, though.

import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 24 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.0

torch.manual_seed(2965)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# nope, this is now a bunch of text from the 50 Victorian novelists dataset, author 26
with open('data/author26.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] if c in stoi else stoi['<unk>'] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# print(encode("hii there"))
# print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
block_size = 8
train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).cuda()
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).cuda()
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')


for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        # print(f"when input is {context.tolist()} the target: {target}")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# check if there's a 'model.pth' in the directory before training the model
if os.path.isfile('model.pth'):
    print("Loading model...")
    model.load_state_dict(torch.load('model.pth'))
else:
    print("Training model...")
    for iter in range(max_iters):
        print('Iteration', iter, '           ', end='\r')
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save the model
    torch.save(model.state_dict(), 'model.pth')

input_text = """
person height and air and i must confess when he dined here the other day there were none to compare with him and we were a party of sixteen luckily there is no distinction of dress nowadays to tell tales but â but â but â yours affectionately i had almost forgot it was s fault he gets into my head more than does me good one very material thing i had to say from henry and myself â i mean about our taking you back into my dear little creature do not stay at to lose your pretty looks those vile sea breezes are the ruin of beauty and health my poor aunt always felt affected if within ten miles of the sea the admiral ol coarse never believed bat i know it was so i am at service and henry s at an s notice i should like the scheme and we make a little and show a in our way and not mind passing through london and seeing the inside of st s square only keep your cousin from me at such a time i not like to be tempted what a long letter â one rd more henry i find has some idea of going into again upon some that you approve but this cannot possibly be permitted before the middle of next week that is he cannot anyhow be spared till after the th for toe have a party that evening the value of a man like henry on such an occasion is what you can have no conception of so you must take it upon my word to be he will see the which i own i am not sorry for â having a little curiosity â and so i think has he though he will not acknowledge it this was a letter to be run through eagerly to be read deliberately to supply matter for much reflection and to leave everything in greater suspense than ever the only certainty to be drawn from it was that nothing decisive had yet taken place had not yet spoken how miss really felt â how she meant to act or might act without or against her meaning â whether his importance to her were quite what it had been the last separation â whether if lessened it were likely to lessen more or to recover itself were subjects for endless conjecture and to be thought of on that day and many days to come without producing any conclusion the idea tha returned the was that miss park after proving herself cooled and staggered by a return to london habits would yet prove herself in the end too much attached to him to give him up she would try to be more ambitious than her heart would allow she would hesitate she would she would condition she would require a great deal but she would finally accept this was s most frequent expectation a house in town â that she thought must be impossible yet there was no saying what miss might not ask the prospect for her cousin grew worse and worse the woman who could speak of him and speak only of his appearance i â what an unworthy attachment to be support from the of mrs she who had known him intimately half a year was ashamed of her those parts of the letter which related only to mr and herself touched her in comparison slightly whether mr went into before or after the th was certainly no concern of hers though everything considered she thought he would go without delay that miss should endeavor to secure a meeting between him and mrs was all in her worst line of conduct and unkind and ill judged but she hoped he would not be by any such degrading curiosity he acknowledged no such and his sister ought to have given him credit for better feelings than her own she was yet more impatient for another letter from town after receiving this than she had been before and for a few days was so unsettled by it park altogether by what had come and what might come that her usual and conversations with were much suspended she could not command her attention as she wished if mr remembered her message to her cousin she thought it very likely most likely that he would write to her at all events â it would be most consistent with his usual kindness and till she got rid of this idea till it gradually wore off by no letters appearing in the course of three or four days more she was in a most restless anxious state at length a something like composure succeeded suspense must be submitted to and must not be allowed to wear her out and make her useless time did something her own exertions something more and she resumed her attentions to and again awakened the same interest in them was growing very fond of her and though without any of the early delight in books which had been so strong in with a disposition much less inclined to pursuits or to information for information s sake she had so strong a desire of not appearing ignorant as with a good clear understanding made her a most attentive profitable thankful pupil was her s explanations and remarks were a most important addition to every essay or every chapter of history what told her of former times dwelt more on her mind than the pages of and she paid her sister the compliment of preferring her style to that of any printed author the early habit of reading was wanting park their conversations however were not always on subjects so high as history or morals others had their hour and of lesser matters none returned so often or remained so long between them as park a description of the people the manners the amusements the ways of 
"""
input_text_2 = """
demand a wholesome of thought dress painting morals and in this they were like the ruling class of any other country particularly of great britain but they differed in being more vigorous and in actually trying to produce the ted standards which all classes everywhere desire but usually despair of ng the longest struggle of the good citizens league was a for the open shop â which was secretly a struggle against all union labor it was an movement with evening classes in en and history and and daily articles in the newspapers so that newly arrived foreigners might learn that the true blue and one hundred per cent american way of settling labor troubles was for workmen to trust and love their the league was more than generous in other which agreed with its aims it helped the y m ca to raise a two hundred thousand dollar fund for a new building and even charles told the spectators at how great an influence for manly christianity the good old y had been in their own lives and the and colonel snow owner of the advocate times was clasping the hand of of the y m ca it is true that afterward when you must come to one of our meetings the ferocious colonel what the hell would i do that for ive got a bar of my own but this did not appear in the public prints the league was of value to the american at a time when certain of the lesser and newspapers were that organization of of the great war one evening a number of young men the burned its records beat the office staff and agreeably out of the window all of the newspapers save the advocate times and the evening advocate attributed this valuable but perhaps hasty direct action to the american then a flying from the good citizens called on the unfair papers and explained that no ex soldier could possibly do such a thing and the saw the light and retained their w en s lone conscientious came home from prison and was run out of town the newspapers referred to the as an mob in all the and triumphs of the good citizens league took part and completely won back to and the affection of his friends but he began to protest i ve done my share in cleaning up the i want to tend to business think just kind of up on this g stuff now he had returned to the church as he had returned to the club he had even endured the lavish greeting which gave him he was worried lest during his late discontent he had his salvation he was not quite sure there was a heaven to be attained but dr john drew said there was and was not going to take a chance one evening when he was walking past dr drew s he went in and foimd the in his study minute â getting call said dr drew in business like tones then to the lo â lot this and reverend drew speaking where the is the proof for next sunday s y ought to have it here well i can t help it if they re all sick i got to have it to night get an a boy and shoot it up here quick he turned without his well brother what c n i do for you i just wanted to ask â tell you how it is here a while ago i guess i got kind of slack took a few drinks and so on what i wanted to ask is how is it if a fellow cuts that all out and comes back to his senses does it sort of well you might say does it score against him in the long run the end dr drew was suddenly and brother â the other things too women no practically you might say practically not at all don t hesitate to tell me brother i that s what i m here for been going on joy rides girls in cars the reverend es no you i ve got a from the don t make a joke association coming to see me in a quarter of an hour and one from the anti birth control union at a quarter of ten he busily at his watch but i can take five minutes off and pray with you kneel right down by your chair brother don t be ashamed to the guidance of god s and he longed to flee but dr drew had down beside his desk and his voice had changed from to an familiarity with sin and with the almighty also knelt while drew o lord thou our brother here who has been led astray by manifold temptations o father make his heart to be pure as e as a little child s let him know again the joy of a manly courage to from evil â came into the study at the sight of the two men he patted on the shoulder and beside him his arm about him he dr drew s with of yes lord help our brother though he was trying to keep his eyes closed between his fingers and saw the glance at his watch as he concluded with a triumphant and let him never be afraid to come to us for counsel and tender care and him know that the church can lead him as a little land dr drew q rang rolled his eyes in the general direction of heaven his watch into his pocket and demanded has the come yet right outside with equal then to brother if it would he i d to fo into the next and pray you while dr drew is receiving the brothers from the don t make a joke association no â no thanks â can t take the time i toward the door thereafter 
"""
input_text_3 = """
argument against the art of it so the criticism h y and the statement that the to was an inspiration whereas the to was but a marvellous piece of delighted him he had always known these things but had not been able to give them expression he wondered how had attained to so clear an understanding and then unconsciously another mind in the argument he said â i wonder what dean thinks of something original i m sure she could not explain that she had not intended to deceive she could not tell him that she was so pressed and by the question of her marriage that she hardly knew what she was saying and had repeated s ideas mechanically she already seemed to stand convicted of he evidently suspected her and all the while he spoke of and she suffered a sort of trembling sickness and that he should have perceived whence her had come her against him suddenly he came to the end of what he had to say their met and he said â very well well be married next week is that soon enough the of his choice fell upon her so suddenly that she answered that next week would do very well she felt that she ought to get up and kiss him and she was painfully conscious that her expression was the reverse of pleased i don t want to limp to the altar were it not for the i d say to morrow but something has happened something has forced you to this he did not dare to suggest scruples of conscience but his thoughts were already back in only that you often have said you d like to marry me one never knows if such things are true it may have been mere gallantry on your part on the other hand i am vain enough to believe that perhaps you meant it then it seemed to her that she must be sincere as i am determined that our present relations shall cease there was no help for it but to come and tell you her eyes were cast down the expression of her face was calm resolution whereas his face betrayed anxiety and the and of the eyes a secret with which he was struggling then i suppose it is scruples of conscience â you ve been to mass at st joseph s we won t enter into that question we ve talked it for the last six years you cannot change me the desire to please was in her and she felt that she had never been so and she was aware that he was showing to better advantage in this scene than she was she wished that he had hesitated if he had only given her some excuse for she did not finish the sentence in her mind but thought instead that she liked him better when he wasn t so good goodness did not seem to suit him she wore a beautiful attractive gown a silk embroidered with silver and he regretted his which kept him from the ball he caught sight of her as she passed down the glittering floor saving with a pretty movement of her shoulders the dress that was slipping from them he saw himself dancing with her they passed in front of a mirror and looking straight over her shoulder his eyes followed the tremulous sparkle of the diamond wings which she wore in her hair then yielding to an impulse of which he was not ashamed for it was as much affection as it was he drew over a chair â he would have knelt at her feet had it not been for his â and passing his arm about her waist he said â dearest i m very fond of you you know that it is not my fault if i prefer to be your lover rather than husband he kissed her on her shoulders laying his cheek on her bosom don t you believe that i am fond of you yes i think you are not a very enthusiastic reply it used to be you who delighted to throw your arms about my neck but all that is over and done with one is not always in such watching each other s eyes they were conscious of their souls every moment it seemed as if their souls must float up and be discovered and while fearing discovery there came a yearning to stand out of all shadow in the full light but they could not tell their souls words fell back and they recognised the mortal lot of and against it he held her face he sought her lips but she turned her face aside leaving him her cheek why do you turn your lips away it is a long time since i ve kissed you you re cold and indifferent lately a memory of shot through her mind and he would have divined her thought if his perception had not been blinded by the passion which swayed him no no we re an engaged couple we re no longer lovers and you think that we should begin by respecting the marriage ceremony she seemed to lose sight of him she perceived only the general idea that outline of her life which he represented and which she could in a way trace in the furniture of the room it was in this room she had said she would be his mistress it was from this room she had started for paris her eyes lighted on the he had bought it in some intention of presenting it to her father some day when they were reconciled the da he had bought for her sake it was the poor little excuse he had devised for coming to see her at she saw the how strange and remote it seemed she looked at the its was an irritation in the there were many books 
"""


# get the probability averaged over the input text
def get_prob_avg(input_text):
    model.eval()

    # Total number of chunks
    total_chunks = len(input_text) // block_size
    if len(input_text) % block_size != 0:
        total_chunks += 1  # Account for the last shorter chunk if any

    avg_probs = []

    # Process each chunk
    for i in range(total_chunks):
        start_index = i * block_size
        end_index = start_index + block_size
        chunk = input_text[start_index:end_index]

        # Encode the chunk
        encoded_chunk = torch.tensor(encode(chunk), dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension

        # Run the model on the chunk
        with torch.no_grad():
            logits, _ = model(encoded_chunk)

        # Calculate probabilities
        probs = F.softmax(logits, dim=-1)  # Apply softmax over the last dimension (vocab_size)

        # Get top-k probabilities for the last token of the chunk
        last_token_probs = probs[0, -1, :]  # Probabilities of the last token
        # top_k_probs, _ = torch.topk(last_token_probs, k)

        # Calculate average of top-k probabilities and store
        # avg_prob = torch.mean(top_k_probs).item()
        avg_probs.append(last_token_probs.max().item())

    # Calculate the overall average probability across all chunks
    overall_avg_prob = sum(avg_probs) / len(avg_probs)
    return overall_avg_prob


print(f"Average probability of author 26 being author 26: {get_prob_avg(input_text):.4f}")
print(f"Average probability of author 42 being author 26: {get_prob_avg(input_text_2):.4f}")
print(f"Average probability of author 15 being author 26: {get_prob_avg(input_text_3):.4f}")

# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(f"Context: {context}")
# idx, prob_avg = m.generate(context, max_new_tokens=2000)
# idx_list = idx[0].tolist()
# print(decode(idx_list))

# X, Y = get_batch(split)
# logits, loss = model(X, Y)
