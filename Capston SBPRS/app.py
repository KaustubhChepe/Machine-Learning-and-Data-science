from flask import Flask,render_template,request
from model import Model 
app = Flask('__name__')
model = Model()
userids = ['joshua','dorothy w','rebecca','walker557','samantha','raeanne','kimmie','cassie','moore222','jds1992','bre234','gordy313']
@app.route('/')
def view():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend_top5():
    print(request.method)
    user_name = request.form['User Name']
    print('User name=',user_name)
    
    if  user_name in userids:
        
            top20_products = model.recommend_products(user_name)
            print(top20_products.head())
            get_top5 = model.top5_products(top20_products)
            return render_template('index.html',tables=[get_top5.to_html(classes='data',header=False,index=False)],text='Recommended products')
    elif not user_name in  userids:
        return render_template('index.html',text='User not Found')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.debug=False
    app.run()