const path = require('path')
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry:{
    app:'./public/js/index.js'
  },
  mode:'development',
  plugins: [new HtmlWebpackPlugin({
    title: 'My App',
    template:'./public/index.html'
  })],
  output:{
    filename:'[name].bundle.js',
    path: path.resolve(__dirname,'dist')
  },
  devServer: {
    contentBase: path.join(__dirname, 'dist'),
    compress: true,
    port: 9000
  },
  module:{
    rules: [
      {
        test: /\.m?js$/,
        exclude: /(node_modules)/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env'],
            plugins: [
              "@babel/plugin-transform-runtime",
              "@babel/plugin-syntax-optional-chaining"
            ]
          }
        }
      }
    ]
  }
}