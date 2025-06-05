const { MongoClient } = require('mongodb');
const fs = require('fs');
const path = require('path');

// Config
const uri = 'mongodb+srv://akashbhatia:OWbmii1dbyi9n7TD@medical-server.bh8ur.mongodb.net/medicalReportsTestDB?retryWrites=true&w=majority';
const client = new MongoClient(uri);
const dbName = 'medicalReportsTestDB';
const collectionName = 'parameters';
const categoryMapPath = path.join(__dirname, 'categories_map.json');
const logPath = path.join(__dirname, 'updated_categories_log.json');

async function updateUnmatchedCategories() {
  try {
    const categoryMap = JSON.parse(fs.readFileSync(categoryMapPath, 'utf-8'));

    await client.connect();
    const db = client.db(dbName);
    const collection = db.collection(collectionName);

    const filter = {
      category: "Unmatched",
      canonicalName: { $in: Object.keys(categoryMap) }
    };

    const docs = await collection.find(filter).toArray();

    if (docs.length === 0) {
      console.log("✅ No unmatched documents found that need updating.");
      return;
    }

    const operations = [];
    const log = [];

    for (const doc of docs) {
      const newCategory = categoryMap[doc.canonicalName];
      operations.push({
        updateOne: {
          filter: { _id: doc._id },
          update: { $set: { category: newCategory } }
        }
      });

      log.push({
        _id: doc._id,
        canonicalName: doc.canonicalName,
        newCategory: newCategory
      });
    }

    const result = await collection.bulkWrite(operations);
    fs.writeFileSync(logPath, JSON.stringify(log, null, 2));

    console.log(`✅ Updated ${result.modifiedCount} documents.`);
    console.log(`📝 Log written to ${logPath}`);
  } catch (err) {
    console.error("❌ Failed to update categories:", err);
  } finally {
    await client.close();
  }
}

updateUnmatchedCategories();
