---
layout: post
title: flutter 食谱 
date: 2024-09-28 09:24:29 +0800
categories:
    - flutter
    - project
---

```dart
import 'package:flutter/material.dart';
import 'data.dart';
import 'models/recipe.dart';

void main() {
  runApp(RecipeApp());
}

class RecipeApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mandy Tan Recipe App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: RecipeListScreen(),
    );
  }
}

class RecipeListScreen extends StatefulWidget {
  @override
  _RecipeListScreenState createState() => _RecipeListScreenState();
}

class _RecipeListScreenState extends State<RecipeListScreen> {
  String selectedCategory = 'All';
  String searchQuery = '';

  @override
  Widget build(BuildContext context) {
    List<Recipe> filteredRecipes = recipes.where((recipe) {
      return (selectedCategory == 'All' ||
              recipe.category == selectedCategory) &&
          (searchQuery.isEmpty ||
              recipe.title.toLowerCase().contains(searchQuery.toLowerCase()));
    }).toList();

    return Scaffold(
      appBar: AppBar(
        title: Text('Mandy Tan Recipes'),
        actions: [
          IconButton(
            icon: Icon(Icons.search),
            onPressed: () {
              showSearch(
                context: context,
                delegate: RecipeSearchDelegate(),
              );
            },
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: DropdownButton<String>(
              value: selectedCategory,
              onChanged: (newValue) {
                setState(() {
                  selectedCategory = newValue!;
                });
              },
              items: <String>[
                'All',
                '豆腐类',
                '蛋类',
                '菜类',
                '鱼类',
                '猪肉类',
                '鸡肉类',
                '牛肉类',
                '汤品类',
                '海鲜类',
                '意大利面类',
                '日式面类',
                '饭类',
                '炒饭类',
                '粥品类',
                '饮品类',
                '甜品类',
                '小吃类'
              ].map<DropdownMenuItem<String>>((String value) {
                return DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                );
              }).toList(),
            ),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: filteredRecipes.length,
              itemBuilder: (context, index) {
                final recipe = filteredRecipes[index];
                return ListTile(
                  leading: Image.network(recipe.imageUrl),
                  title: Text(recipe.title),
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) =>
                            RecipeDetailScreen(recipe: recipe),
                      ),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class RecipeDetailScreen extends StatelessWidget {
  final Recipe recipe;

  RecipeDetailScreen({required this.recipe});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(recipe.title),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: ListView(
          children: [
            Image.network(recipe.imageUrl),
            SizedBox(height: 16.0),
            Text(
              recipe.title,
              style: TextStyle(fontSize: 24.0, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8.0),
            Text(recipe.description),
            SizedBox(height: 16.0),
            Text(
              '材料 : ',
              style: TextStyle(fontSize: 20.0, fontWeight: FontWeight.bold),
            ),
            ...recipe.ingredients.map((ingredient) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4.0),
                  child: Text(
                    ingredient,
                    style: TextStyle(fontSize: 16.0),
                  ),
                )),
            SizedBox(height: 16.0),
            Text(
              '步骤 : ',
              style: TextStyle(fontSize: 20.0, fontWeight: FontWeight.bold),
            ),
            ...recipe.steps.map((step) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4.0),
                  child: Text(
                    step,
                    style: TextStyle(fontSize: 16.0),
                  ),
                )),
          ],
        ),
      ),
    );
  }
}

class RecipeSearchDelegate extends SearchDelegate {
  @override
  List<Widget> buildActions(BuildContext context) {
    return [
      IconButton(
        icon: Icon(Icons.clear),
        onPressed: () {
          query = '';
        },
      ),
    ];
  }

  @override
  Widget buildLeading(BuildContext context) {
    return IconButton(
      icon: AnimatedIcon(
        icon: AnimatedIcons.menu_arrow,
        progress: transitionAnimation,
      ),
      onPressed: () {
        close(context, null);
      },
    );
  }

  @override
  Widget buildResults(BuildContext context) {
    List<Recipe> searchResults = recipes.where((recipe) {
      return recipe.title.toLowerCase().contains(query.toLowerCase());
    }).toList();

    return ListView.builder(
      itemCount: searchResults.length,
      itemBuilder: (context, index) {
        final recipe = searchResults[index];
        return ListTile(
          leading: Image.network(recipe.imageUrl),
          title: Text(recipe.title),
          onTap: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => RecipeDetailScreen(recipe: recipe),
              ),
            );
          },
        );
      },
    );
  }

  @override
  Widget buildSuggestions(BuildContext context) {
    List<Recipe> searchSuggestions = recipes.where((recipe) {
      return recipe.title.toLowerCase().contains(query.toLowerCase());
    }).toList();

    return ListView.builder(
      itemCount: searchSuggestions.length,
      itemBuilder: (context, index) {
        final recipe = searchSuggestions[index];
        return ListTile(
          leading: Image.network(recipe.imageUrl),
          title: Text(recipe.title),
          onTap: () {
            query = recipe.title;
            showResults(context);
          },
        );
      },
    );
  }
}
```