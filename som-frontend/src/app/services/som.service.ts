import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SomService {
  private apiUrl = 'http://localhost:5000';

  constructor(private http: HttpClient) { }

  trainSom(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train-som`, {});
  }

  getImage(): Observable<any> {
    return this.http.get(`${this.apiUrl}/get-image`, { responseType: 'blob' });
  }
}
