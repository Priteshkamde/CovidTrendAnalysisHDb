# Generated by Django 2.1.1 on 2020-05-26 11:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Operations',
            fields=[
                ('email', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('operations', models.CharField(max_length=1000)),
            ],
        ),
    ]
