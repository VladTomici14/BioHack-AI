const spawner = require('child_process').spawn;

const data_to_pass_in = "Send this to your python script";

const bird_link = "https://static.scientificamerican.com/sciam/cache/file/7A715AD8-449D-4B5A-ABA2C5D92D9B5A21_source.png";

const python_process = spawner('python', ['python.py', JSON.stringify(bird_link)]);

python_process.stdout.on('data', (data) => {
    console.log('data received from python script:', JSON.parse(data.toString()))
});